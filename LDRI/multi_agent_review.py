"""Multi-agent reward review pipeline for LDRI.

Architecture (six-priority multi-agentic workflow):
  context_builder → diagnostic_agent (Python)
  → [soc_safety | fuel_efficiency | reward_hacking | fc_durability | battery_usage] (LLM x5)
  + scale_reviewer (Python) + trainability_scheduler (Python)
  → tradeoff_mediator (LLM) → orchestrator (LLM x1)
  → counterfactual_evaluator (Python) → proposer (LLM x1..N) → TPE/aligned-gate verifier

Priority 1: Training Diagnostic Agent (Python — pre-review failure classification)
Priority 2: Specialist Agents — fc_durability and battery_usage with fuel-efficiency-first personas
Priority 3: Trade-off Mediator — resolves reviewer conflicts; fuel efficiency > SOC tightening
Priority 4: Counterfactual Evaluator — flags high-risk orchestrator_spec changes
Priority 5: Iteration Memory — trajectory_memory + insight_memory passed via iteration_memory kwarg
Priority 6: Policy Quality — Python metric score computed externally in main_ldri_refine

CRITICAL: Fuel efficiency (g/km, h2_100km) is ALWAYS the primary objective.
          SOC is a constraint to ENABLE fuel efficiency, not an end in itself.

Entry point: run_multi_agent_review()
Returns the same dict shape as _run_llm_with_tpe_retry().

Risk precedence (fuel efficiency is PRIMARY):
  soc_safety hard_reject > reward_hacking hard_reject > scale_reviewer hard_reject
  > fuel_efficiency (PRIMARY OBJECTIVE) > soc_safety soft_revise > trainability_scheduler

Per-agent model selection:
  Provider-specific tiers are defined in llm_presets.AGENT_MODELS.
  scale_reviewer, trainability_scheduler, diagnostic_agent, counterfactual_evaluator
  are pure Python (no LLM calls).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from feedback_parser import parse_train_log
from ldri_observer import _write_json, _write_text, infer_key_weight_changes
from llm_presets import get_agent_model
from reward_lr_tuner import _training_trend_summary
from reward_patcher import ensure_logging_contract, extract_method_from_llm_response
from ems_tools import ToolContext, execute_tool, get_tools_for_agent


REQUIRED_INFO_KEYS = [
    "EMS_reward",
    "h2_fcs",
    "h2_batt",
    "h2_equal",
    "soc_cost",
    "h2_cost",
    "fcs_soh_cost",
    "batt_soh_cost",
    "objective_cost",
    "soc_in_bounds",
]

# Absolute coefficient caps derived from iter_004 of the reference good run
# (k_oob=3.0, w_soc_eff=0.78x, w_h2_eff=1.65x → best_reward=-9636 at ep136/500).
# soft: warn and request correction.  hard: block and force prescribed value.
_SCALE_CAPS: Dict[str, Dict[str, float]] = {
    "k_oob":            {"soft": 5.0,  "hard": 8.0},
    "k_inbounds":       {"soft": 3.0,  "hard": 5.0},
    "w_soc_multiplier": {"soft": 1.5,  "hard": 2.5},
    "w_h2_multiplier":  {"soft": 2.5,  "hard": 4.0},
}
# Per-iteration change-rate caps (relative to previous iteration's value).
_SCALE_CHANGE_SOFT = 1.00   # >100 % change → soft_revise
_SCALE_CHANGE_HARD = 2.00   # >200 % change → hard_reject


# ─────────────────────────────────────────────────────────────────────────────
# Historical reward guard helpers
# ─────────────────────────────────────────────────────────────────────────────

def _canonicalize_method_source(method_source: str) -> str:
    text = textwrap.dedent(str(method_source or "")).strip()
    if not text:
        return ""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()

    fn_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn_node = node
            break
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    fn_node = sub
                    break
        if fn_node is not None:
            break

    if fn_node is None:
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()

    return ast.dump(fn_node, annotate_fields=True, include_attributes=False)


def _method_signature(method_source: str) -> str:
    canonical = _canonicalize_method_source(method_source)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _origin_sort_key(origin: str):
    if origin == "bootstrap":
        return (-1, origin)
    m = re.fullmatch(r"iter_(\d+)", str(origin))
    if m:
        return (int(m.group(1)), origin)
    m = re.fullmatch(r"current_iter_(\d+)_pre_refine", str(origin))
    if m:
        return (int(m.group(1)), origin)
    return (10**9, str(origin))


def _history_match_label(entry: Dict[str, Any]) -> str:
    origins = entry.get("origins", [])
    if not isinstance(origins, list) or not origins:
        return entry.get("signature", "unknown")[:12]
    ordered = sorted((str(origin) for origin in origins), key=_origin_sort_key)
    return ", ".join(ordered)


def _build_historical_reward_guard_payload(
    reward_history: Optional[Dict[str, Dict[str, Any]]],
    current_method: str,
) -> Dict[str, Any]:
    history = reward_history or {}
    current_signature = _method_signature(current_method)

    historical_labels: List[str] = []
    for signature, entry in history.items():
        if signature == current_signature:
            continue
        label = _history_match_label(entry)
        if label:
            historical_labels.append(label)

    seen = set()
    ordered_labels: List[str] = []
    for label in sorted(historical_labels, key=_origin_sort_key):
        if label in seen:
            continue
        seen.add(label)
        ordered_labels.append(label)

    shown = ordered_labels[:8]
    return {
        "current_signature_prefix": current_signature[:12],
        "rules": [
            "Returning the current reward unchanged is invalid because it has already been trained.",
            "Returning any previously trained reward variant is invalid even if it improves instantaneous TPE delta.",
            "Exact structural matches to already-trained get_reward(self) variants are rejected before TPE evaluation.",
            "When a recent reward looks wrong, fix it with a new local refinement instead of reverting to an older trained reward.",
        ],
        "historical_versions": shown,
        "omitted_count": max(0, len(ordered_labels) - len(shown)),
    }


def _render_historical_reward_guard(guard: Dict[str, Any]) -> str:
    return json.dumps(guard, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Scale reviewer: pure Python (no LLM)
# Extracts numeric scaling coefficients from reward source and checks them
# against _SCALE_CAPS and per-iteration change-rate limits.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_scale_coefficients(source: str) -> Dict[str, Optional[float]]:
    """Extract named numeric scaling coefficients from reward source code."""
    coeffs: Dict[str, Optional[float]] = {}

    # k_oob / k_inbounds — always a direct float assignment
    for name in ("k_oob", "k_inbounds"):
        m = re.search(rf"\b{name}\s*=\s*([\d]+(?:\.[\d]*)?)", source)
        coeffs[name] = float(m.group(1)) if m else None

    # h2 effective multiplier — three common patterns:
    #   Pattern A (named scale var):  h2_scale = 8.0
    #   Pattern B (inline left):      w_h2_eff = 1.65 * float(self.w_h2)
    #   Pattern C (inline right):     w_h2_eff = float(self.w_h2) * 1.65
    h2_mult: Optional[float] = None
    m = re.search(r"\bh2_scale\s*=\s*([\d]+(?:\.[\d]*)?)", source)
    if m:
        h2_mult = float(m.group(1))
    if h2_mult is None:
        m = re.search(r"\bw_h2_eff\s*=\s*([\d]+(?:\.[\d]*)?)\s*\*", source)
        if m:
            h2_mult = float(m.group(1))
    if h2_mult is None:
        m = re.search(
            r"\bw_h2_eff\s*=\s*float\(self\.w_h2\)\s*\*\s*([\d]+(?:\.[\d]*)?)", source
        )
        if m:
            h2_mult = float(m.group(1))
    coeffs["w_h2_multiplier"] = h2_mult

    # soc effective multiplier — three common patterns:
    #   Pattern A (inline left):   w_soc_eff = 0.78 * float(self.w_soc)
    #   Pattern B (inline right):  w_soc_eff = float(self.w_soc) * 0.78
    #   Pattern C (named var):     soc_scale = 1.25
    soc_mult: Optional[float] = None
    m = re.search(r"\bw_soc_eff\s*=\s*([\d]+(?:\.[\d]*)?)\s*\*", source)
    if m:
        soc_mult = float(m.group(1))
    if soc_mult is None:
        m = re.search(
            r"\bw_soc_eff\s*=\s*float\(self\.w_soc\)\s*\*\s*([\d]+(?:\.[\d]*)?)", source
        )
        if m:
            soc_mult = float(m.group(1))
    if soc_mult is None:
        m = re.search(r"\bsoc_scale\s*=\s*([\d]+(?:\.[\d]*)?)", source)
        if m:
            soc_mult = float(m.group(1))
    coeffs["w_soc_multiplier"] = soc_mult

    return coeffs


def _compute_scale_check_from_packet(
    review_packet: Dict[str, Any],
    prev_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect coefficient scale violations; returns a reviewer-compatible result dict."""
    current_source = review_packet.get("reward_source", "")
    current_coeffs = _extract_scale_coefficients(current_source)
    prev_coeffs = _extract_scale_coefficients(prev_source) if prev_source else {}

    violations: List[Dict[str, Any]] = []
    decision = "pass"

    def _escalate(sev: str) -> None:
        nonlocal decision
        if sev == "hard_reject":
            decision = "hard_reject"
        elif sev == "soft_revise" and decision == "pass":
            decision = "soft_revise"

    # ── Absolute cap checks ───────────────────────────────────────────────────
    for name, caps in _SCALE_CAPS.items():
        val = current_coeffs.get(name)
        if val is None:
            continue
        if val >= caps["hard"]:
            prescribed = round(caps["soft"] * 0.75, 3)
            violations.append({
                "coeff": name, "current": val,
                "cap_violated": f"hard (>={caps['hard']})",
                "severity": "hard_reject", "prescribed": prescribed,
            })
            _escalate("hard_reject")
        elif val >= caps["soft"]:
            prescribed = round(caps["soft"] * 0.85, 3)
            violations.append({
                "coeff": name, "current": val,
                "cap_violated": f"soft (>={caps['soft']})",
                "severity": "soft_revise", "prescribed": prescribed,
            })
            _escalate("soft_revise")

    # ── Per-iteration change-rate checks ─────────────────────────────────────
    for name, curr_val in current_coeffs.items():
        if curr_val is None:
            continue
        prev_val = prev_coeffs.get(name)
        if prev_val is None or prev_val == 0.0:
            continue
        ratio = (curr_val - prev_val) / abs(prev_val)
        if ratio > _SCALE_CHANGE_HARD:
            prescribed = round(prev_val * (1.0 + _SCALE_CHANGE_SOFT), 3)
            violations.append({
                "coeff": name, "prev": prev_val, "current": curr_val,
                "change_ratio": round(ratio, 2),
                "cap_violated": f"change_rate hard (>{_SCALE_CHANGE_HARD:.0%})",
                "severity": "hard_reject", "prescribed": prescribed,
            })
            _escalate("hard_reject")
        elif ratio > _SCALE_CHANGE_SOFT:
            prescribed = round(prev_val * (1.0 + _SCALE_CHANGE_SOFT), 3)
            violations.append({
                "coeff": name, "prev": prev_val, "current": curr_val,
                "change_ratio": round(ratio, 2),
                "cap_violated": f"change_rate soft (>{_SCALE_CHANGE_SOFT:.0%})",
                "severity": "soft_revise", "prescribed": prescribed,
            })
            _escalate("soft_revise")

    prescribed_corrections = {
        v["coeff"]: v["prescribed"]
        for v in violations
        if "prescribed" in v
    }

    risk_score = 0.9 if decision == "hard_reject" else (0.5 if decision == "soft_revise" else 0.0)

    return {
        "agent": "scale_reviewer",
        "decision": decision,
        "risk_score": round(risk_score, 3),
        "confidence": 1.0,
        "current_coefficients": {k: v for k, v in current_coeffs.items() if v is not None},
        "violations": violations,
        "prescribed_corrections": prescribed_corrections,
        "_source": "python_computation",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Priority 1: Training Diagnostic Agent (pure Python, pre-review)
# Classifies failure mode before LLM reviewers so they share a common context.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_diagnostic_from_packet(review_packet: Dict[str, Any]) -> Dict[str, Any]:
    """Classify training failure mode; returns reviewer_hints injected into each reviewer's packet. 
    --> but still hints are hard-coded"""
    h2_stats = review_packet.get("h2_stats", {})
    soc_stats = review_packet.get("soc_stats", {})
    trend = review_packet.get("training_trend", {})
    recent_eps = review_packet.get("recent_episodes", [])

    total_eps = max(1, int(soc_stats.get("total_episodes") or 1))
    oob_count = int(soc_stats.get("out_of_bounds_episodes") or 0)
    oob_rate = oob_count / total_eps

    h2_mean = h2_stats.get("h2_100km_mean")
    h2_min = h2_stats.get("h2_100km_min")
    h2_max = h2_stats.get("h2_100km_max")

    best_ep_ratio = float(trend.get("best_episode_ratio") or 1.0)
    tail_gap = float(trend.get("tail_gap_ratio") or 0.0)
    trend_factor = float(trend.get("trend_factor") or 1.0)

    soc_swings = []
    for ep in recent_eps:
        mn = ep.get("soc_min")
        mx = ep.get("soc_max")
        if mn is not None and mx is not None:
            soc_swings.append(float(mx) - float(mn))
    mean_swing = sum(soc_swings) / len(soc_swings) if soc_swings else None

    # Failure mode classification
    if oob_rate > 0.20:
        failure_mode = "soc_spiral"
    elif mean_swing is not None and mean_swing < 0.05 and h2_mean is not None and h2_mean > 2.5:
        failure_mode = "h2_suppression"
    elif best_ep_ratio < 0.3 and tail_gap > 0.08:
        failure_mode = "reward_collapse"
    elif trend_factor > 0.85 and oob_rate < 0.05:
        failure_mode = "healthy"
    else:
        failure_mode = "mixed"

    convergence = (
        "good" if trend_factor >= 0.85 and tail_gap < 0.05
        else ("degraded" if trend_factor >= 0.65 else "failed")
    )

    if h2_mean is not None and h2_min is not None:
        h2_range = (h2_max or h2_mean) - h2_min
        h2_trend = "improving" if h2_range > 0.5 else ("stable" if h2_mean < 2.0 else "regressing")
    else:
        h2_trend = "no_data"

    if oob_rate > 0.10:
        soc_stability = "drifting"
    elif mean_swing is not None and mean_swing < 0.05:
        soc_stability = "locked_near_target"
    elif mean_swing is not None and mean_swing > 0.30:
        soc_stability = "oscillating"
    else:
        soc_stability = "stable"

    swing_str = f"{mean_swing:.3f}" if mean_swing is not None else "N/A"
    # hard-coded hints based on failure mode; these are read by the LLM reviewers to focus their analysis and recommendations
    hints: Dict[str, str] = {}
    if failure_mode == "soc_spiral":
        hints = {
            "soc_safety": f"OOB rate is high ({oob_rate:.1%}); check k_oob and w_soc strength",
            "fuel_efficiency": "SOC instability may be suppressing battery buffering; assess h2 impact",
            "reward_hacking": "Check if OOB penalty is exponential or runaway; scale violation likely",
            "fc_durability": "SOC instability may force FCS to extreme operating points",
            "battery_usage": f"OOB rate {oob_rate:.1%}; battery may be overdriven rather than buffering",
        }
    elif failure_mode == "h2_suppression":
        hints = {
            "soc_safety": f"SOC swing is very small ({swing_str}); battery NOT buffering — this is a fuel problem",
            "fuel_efficiency": f"CRITICAL: battery buffering suppressed (swing={swing_str}) → FCS follows load → H2 waste. ROOT CAUSE. Reduce w_soc.",
            "reward_hacking": "Check if soc_cost complexity is suppressing the h2 gradient signal",
            "fc_durability": f"Low SOC swing ({swing_str}) may mean FCS is idling/cycling on low-speed segments",
            "battery_usage": f"CRITICAL: mean SOC swing={swing_str} indicates battery is NOT buffering; this is the root cause of H2 waste",
        }
    elif failure_mode == "reward_collapse":
        hints = {
            "soc_safety": "Best reward appeared early and decayed; reward may be collapsing",
            "fuel_efficiency": "Reward collapse often from h2_cost dominating before SOC stabilized; check balance",
            "reward_hacking": "Check for scale blow-up or exploitable terms causing early-episode reward spike",
            "fc_durability": "Convergence failure; FCS operating quality data may be unreliable",
            "battery_usage": "Convergence failure; battery utilization data may be unreliable",
        }
    else:
        hints = {
            "soc_safety": "No critical SOC pattern detected",
            "fuel_efficiency": "Assess h2 trend vs SOC balance for continued improvement",
            "reward_hacking": "Standard over-shaping check",
            "fc_durability": "Assess FCS operating quality for idling/cycling patterns",
            "battery_usage": "Verify battery is actively buffering FCS load",
        }

    return {
        "agent": "training_diagnostic",
        "failure_mode": failure_mode,
        "convergence_quality": convergence,
        "h2_trend_diagnosis": h2_trend,
        "soc_stability_diagnosis": soc_stability,
        "oob_rate": round(oob_rate, 4),
        "mean_soc_swing": round(mean_swing, 4) if mean_swing is not None else None,
        "best_episode_ratio": round(best_ep_ratio, 4),
        "trend_factor": round(trend_factor, 4),
        "reviewer_hints": hints,
        "diagnostic_summary": (
            f"Failure mode: {failure_mode}. Convergence: {convergence}. "
            f"H2 trend: {h2_trend}. SOC stability: {soc_stability}. "
            f"OOB rate: {oob_rate:.1%}, mean SOC swing: {swing_str}."
        ),
        "_source": "python_computation",
    }


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract first JSON object from ```json...``` block or bare object."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Greedy match from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"no JSON object found in LLM response (first 400 chars): {text[:400]}")


def _llm_usage_totals(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "has_unpriced_calls": False,
    }
    for rec in records:
        usage = rec.get("usage") or {}
        cost = rec.get("cost") or {}
        totals["input_tokens"] += int(usage.get("input_tokens", usage.get("prompt_tokens")) or 0)
        totals["output_tokens"] += int(usage.get("output_tokens", usage.get("completion_tokens")) or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or cost.get("total_tokens") or 0)
        if cost.get("total_cost") is None:
            totals["has_unpriced_calls"] = True
        else:
            totals["total_cost_usd"] += float(cost.get("total_cost") or 0.0)
    return totals


def _write_llm_usage_log(path: Path, records: List[Dict[str, Any]]) -> None:
    _write_json(path, {
        "records": records,
        "totals": _llm_usage_totals(records),
        "pricing_note": (
            "Set LLM_PRICE_USD_PER_1M_JSON to estimate USD cost, e.g. "
            "'{\"provider:model\":{\"input\":3,\"output\":15}}'. "
            "Rates are USD per 1M tokens."
        ),
    })


def _tracked_complete(
    *,
    client,
    usage_records: List[Dict[str, Any]],
    usage_path: Path,
    stage: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    model: Optional[str] = None,
) -> str:
    if hasattr(client, "complete_with_usage"):
        completion = client.complete_with_usage(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            model=model,
        )
        text = completion.text
        record = {
            "stage": stage,
            "provider": completion.provider,
            "model": completion.model,
            "requested_model": model if model is not None else getattr(client, "model", None),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_chars": len(prompt),
            "response_chars": len(text),
            "usage": completion.usage,
            "cost": completion.cost,
        }
    else:
        text = client.complete(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            model=model,
        )
        record = {
            "stage": stage,
            "provider": getattr(client, "provider", None),
            "model": model if model is not None else getattr(client, "model", None),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_chars": len(prompt),
            "response_chars": len(text),
            "usage": {},
            "cost": {"currency": "USD", "total_cost": None, "pricing_source": None},
            "note": "client does not expose complete_with_usage",
        }
    usage_records.append(record)
    _write_llm_usage_log(usage_path, usage_records)
    return text


def _tracked_complete_with_tools(
    *,
    client,
    usage_records: List[Dict[str, Any]],
    usage_path: Path,
    stage: str,
    prompt: str,
    tools: List[Dict[str, Any]],
    tool_executor,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    model: Optional[str] = None,
    ctx: Optional[ToolContext] = None,
) -> str:
    """Like _tracked_complete but uses Anthropic tool_use when ctx is available.

    Falls back to _tracked_complete for non-Anthropic providers or when ctx is None.
    """
    use_tools = (
        ctx is not None
        and bool(tools)
        and getattr(client, "provider", "") == "anthropic"
        and hasattr(client, "complete_with_tools")
    )
    if not use_tools:
        return _tracked_complete(
            client=client,
            usage_records=usage_records,
            usage_path=usage_path,
            stage=stage,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            model=model,
        )

    completion = client.complete_with_tools(
        prompt=prompt,
        tools=tools,
        tool_executor=tool_executor,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        model=model,
    )
    record = {
        "stage":            stage,
        "provider":         completion.provider,
        "model":            completion.model,
        "requested_model":  model if model is not None else getattr(client, "model", None),
        "temperature":      temperature,
        "max_tokens":       max_tokens,
        "prompt_chars":     len(prompt),
        "response_chars":   len(completion.text),
        "usage":            completion.usage,
        "cost":             completion.cost,
        "note":             "tool_use",
    }
    usage_records.append(record)
    _write_llm_usage_log(usage_path, usage_records)
    return completion.text


# ─────────────────────────────────────────────────────────────────────────────
# Context builder (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def build_review_packet(
    *,
    feedback_bundle: Dict[str, str],
    current_method: str,
    chunk_result,
    args,
    iteration: int,
    iteration_memory: Optional[Dict[str, Any]] = None,
    policy_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate all training artifacts into a normalized review_packet.

    No LLM call. Pure data aggregation from train_log + args + feedback.
    This is the shared, normalized input for all reviewers.
    iteration_memory: {trajectory: {...}, insights: [...]} from main loop (Priority 5).
    policy_quality: Python policy score computed after training (Priority 6).
    """
    episodes = parse_train_log(chunk_result.log_path)

    soc_low = float(getattr(args, "soc_low", 0.4))
    soc_high = float(getattr(args, "soc_high", 0.8))
    soc_eps = [ep for ep in episodes if ep.soc_min is not None]
    out_of_bounds_count = sum(
        1 for ep in soc_eps
        if (ep.soc_min is not None and ep.soc_min < soc_low)
        or (ep.soc_max is not None and ep.soc_max > soc_high)
    )

    h2_eps = [ep for ep in episodes if ep.h2_100km is not None]
    h2_vals = [ep.h2_100km for ep in h2_eps]

    try:
        trend = _training_trend_summary(chunk_result.log_path, args)
    except Exception:
        trend = {"best_episode_ratio": 1.0, "tail_gap_ratio": 0.0, "trend_factor": 1.0}

    n_recent = max(5, int(getattr(args, "feedback_process_episodes", 10) or 10))
    recent_summaries: List[Dict[str, Any]] = []
    for ep in episodes[-n_recent:]:
        s: Dict[str, Any] = {"episode": ep.episode}
        for attr, key in [
            ("soc_min", "soc_min"), ("soc_max", "soc_max"), ("soc_mean", "soc_mean"),
            ("in_bounds_rate", "in_bounds_rate_pct"), ("h2_100km", "h2_100km"),
            ("ep_cumulative_r", "ep_cumulative_r"), ("mean_h2_cost", "mean_h2_cost"),
            ("mean_soc_cost", "mean_soc_cost"), ("mean_fcs_cost", "mean_fcs_cost"),
            ("mean_batt_cost", "mean_batt_cost"), ("mean_objective_cost", "mean_objective_cost"),
        ]:
            v = getattr(ep, attr, None)
            if v is not None:
                s[key] = v
        recent_summaries.append(s)

    reward_eps = [ep for ep in episodes if ep.ep_cumulative_r is not None]
    best_ep = max(reward_eps, key=lambda e: e.ep_cumulative_r, default=None)
    worst_ep = min(reward_eps, key=lambda e: e.ep_cumulative_r, default=None)

    return {
        "iteration": iteration,
        "chunk": {
            "start_episode": chunk_result.start_episode,
            "end_episode": chunk_result.end_episode,
        },
        "soc_bounds": {"low": soc_low, "high": soc_high},
        "soc_stats": {
            "total_episodes": len(episodes),
            "episodes_with_soc": len(soc_eps),
            "out_of_bounds_episodes": out_of_bounds_count,
            "global_soc_min": min((ep.soc_min for ep in soc_eps), default=None),
            "global_soc_max": max((ep.soc_max for ep in soc_eps if ep.soc_max is not None), default=None),
        },
        "h2_stats": {
            "episodes_with_h2": len(h2_eps),
            "h2_100km_mean": sum(h2_vals) / len(h2_vals) if h2_vals else None,
            "h2_100km_min": min(h2_vals, default=None),
            "h2_100km_max": max(h2_vals, default=None),
        },
        "training_trend": trend,
        "best_episode": {
            "episode": best_ep.episode if best_ep else None,
            "ep_cumulative_r": best_ep.ep_cumulative_r if best_ep else None,
        },
        "worst_episode": {
            "episode": worst_ep.episode if worst_ep else None,
            "ep_cumulative_r": worst_ep.ep_cumulative_r if worst_ep else None,
        },
        "recent_episodes": recent_summaries,
        "lr_settings": {
            "lr_critic": float(getattr(args, "lr_critic", 7e-4)),
            "lr_critic_min": float(getattr(args, "lr_critic_min", 2e-4)),
            "lr_actor": float(getattr(args, "lr_actor", 3e-4)),
            "lr_actor_min": float(getattr(args, "lr_actor_min", 3e-5)),
            "lr_alpha": float(getattr(args, "lr_alpha", 5e-4)),
            "lr_warmup_ratio": float(getattr(args, "lr_warmup_ratio", 0.10)),
        },
        "reward_source": current_method,
        "feedback": {
            "process": feedback_bundle["PROCESS_FEEDBACK"],
            "trajectory": feedback_bundle["TRAJECTORY_FEEDBACK"],
            "preference": feedback_bundle["PREFERENCE_FEEDBACK"],
        },
        "prior_iterations_summary": iteration_memory or {},
        "policy_quality": policy_quality or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reviewer prompts (each reviewer receives a focused slice of the packet)
# ─────────────────────────────────────────────────────────────────────────────

def _soc_prompt(packet: Dict[str, Any]) -> str:
    focused = {
        "iteration": packet["iteration"],
        "soc_bounds": packet["soc_bounds"],
        "soc_stats": packet["soc_stats"],
        "recent_episodes": [
            {k: ep[k] for k in ("episode", "soc_min", "soc_max", "in_bounds_rate_pct") if k in ep}
            for ep in packet["recent_episodes"]
        ],
        "reward_source": packet["reward_source"],
    }
    return (
        "You are the SOC safety reviewer for an FCEV Energy Management System reward function.\n"
        "Your ONLY responsibility: assess SOC safety risk.\n\n"
        "Focus on:\n"
        "- soc_stats.out_of_bounds_episodes / total_episodes ratio\n"
        "- recent_episodes: soc_min, soc_max, in_bounds_rate_pct\n"
        "- reward_source: presence of explicit quadratic SOC target anchor AND explicit out-of-bounds strengthening\n\n"
        "SOC variation interpretation (read this before judging):\n"
        "- Safe operating range: [soc_low=0.4, soc_high=0.8]. OOB episodes are this reviewer's primary concern.\n"
        "- SOC swing WITHIN bounds (soc_max - soc_min = 0.20–0.40) is DESIRABLE and ENCOURAGED: the battery\n"
        "  is buffering FCS load, which directly improves g/km fuel efficiency. Do NOT flag healthy in-bounds\n"
        "  swing as a problem or propose tighter tracking to reduce it.\n"
        "- Consistently small swing (< 0.20) is a fuel efficiency risk, not a safety benefit — flag via\n"
        "  soc_swing_ok=false (pass-through field read by the orchestrator) but do NOT raise SOC safety concerns.\n\n"
        "Hard constraints you must enforce:\n"
        "- soc_cost base term MUST be one of these two allowed quadratic forms:\n"
        "    (a) Pure quadratic:     w_soc * (soc - soc_target)^2\n"
        "    (b) Deadband quadratic: w_soc * k_inbounds * max(0, |soc - soc_target| - db)^2\n"
        "        db ∈ [0.02, 0.08] — penalty-free zone near target; a wider deadband is PREFERRED to allow\n"
        "        battery buffering without unnecessary SOC restoration force\n"
        "- If out-of-bounds strengthening is used, keep that strengthening quadratic (dist^2 form), NOT exponential or higher-order\n"
        "- If OOB episodes are <= 5% and recent in_bounds_rate_pct is >= 95%, do NOT propose increasing w_soc, "
        "k_inbounds, or k_oob; SOC is already stable enough and extra SOC pressure can suppress fuel economy.\n"
        "- Do NOT replace quadratic SOC penalties with linear or absolute-deviation penalties\n"
        "- soc_cost must remain non-negative and finite\n"
        "- Do NOT introduce progress-aware, terminal-aware, or non-causal SOC logic\n"
        "- Replay-compatible only: do not rely on step, episode_steps, progress, travel, or terminal flags\n\n"
        "SOC complexity budget (CRITICAL — enforce this strictly):\n"
        "- soc_cost is allowed AT MOST 3 distinct additive terms:\n"
        "    (1) base quadratic target-tracking: form (a) or (b) above  [REQUIRED]\n"
        "    (2) out-of-bounds quadratic strengthening: w_soc * k_oob * (dist_low^2 + dist_high^2)  [optional]\n"
        "    (3) one optional recovery shaping term, bounded and causal  [only if OOB episodes > 5%]\n"
        "- Do NOT propose exponential barriers, margin zones, danger zones, caution/critical/warn bands, "
        "proximity gates, or any other in-bounds penalty layer beyond the base quadratic\n"
        "- If soc_cost already contains more than 3 terms, flag this as over-shaping and recommend simplification\n\n"
        "PASS graduation criteria (return 'pass' when ALL of these hold):\n"
        "- out_of_bounds_episodes / total_episodes <= 0.05 (i.e., <= 5%)\n"
        "- All recent_episodes show in_bounds_rate_pct >= 95.0\n"
        "- reward_source contains a valid base quadratic SOC target-tracking term (form (a) or (b))\n"
        "- reward_source does NOT violate any hard constraints above\n"
        "When these criteria are met, return 'pass' — do NOT request more SOC penalty layers, "
        "and do NOT propose increasing w_soc, k_inbounds, or k_oob.\n\n"
        "soft_revise criteria (only when pass criteria are NOT met):\n"
        "- out_of_bounds_episodes / total_episodes > 5%, OR recent in_bounds_rate_pct < 95%\n"
        "- In this case, propose ONLY coefficient adjustment (increase k_oob) or, if SOC drift is\n"
        "  clearly above bounds, a moderate increase to w_soc or k_inbounds\n"
        "- If OOB is low but per-episode SOC swing is very small (< 0.20): do NOT tighten soc_cost further;\n"
        "  instead flag to the orchestrator that battery buffering may be suppressed\n"
        "- Do NOT propose adding new penalty layers; coefficient tuning is always sufficient\n"
        "- Coefficient tuning for stable in-bounds SOC must relax SOC pressure, not increase it.\n\n"
        "hard_reject criteria:\n"
        "- reward_source removes the quadratic SOC target anchor entirely\n"
        "- replaces main SOC term with linear or absolute-deviation penalty\n"
        "- makes soc_cost potentially negative\n"
        "- introduces progress/terminal-aware SOC shaping\n"
        "- contains non-quadratic OOB terms (exponential, polynomial > degree 2) that dominate over the base quadratic\n\n"
        "Output ONLY a compact JSON object inside ```json ... ``` with schema:\n"
        "{\n"
        '  "agent": "soc_safety_reviewer",\n'
        '  "decision": "pass | soft_revise | hard_reject",\n'
        '  "risk_score": <0.0-1.0>,\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "summary": "<one sentence: key metric values and decision reason>",\n'
        '  "soc_term_count": <int: number of distinct additive terms in soc_cost>,\n'
        '  "over_shaping": <bool: true if soc_term_count > 3>,\n'
        '  "soc_swing_ok": <bool: true if per-episode soc_max - soc_min is >= 0.05 in most recent episodes>,\n'
        '  "proposals": [{"target": <str>, "direction": "increase_small|decrease_small|keep|remove", "reason": <str>}]\n'
        "}\n\n"
        "REVIEW_PACKET:\n"
        + json.dumps(focused, ensure_ascii=False, indent=2)
    )


def _fuel_prompt(packet: Dict[str, Any]) -> str:
    focused = {
        "iteration": packet["iteration"],
        "h2_stats": packet["h2_stats"],
        "recent_episodes": [
            {k: ep[k] for k in ("episode", "h2_100km", "mean_h2_cost", "mean_soc_cost",
                                 "mean_fcs_cost", "mean_batt_cost",
                                 "mean_objective_cost", "in_bounds_rate_pct",
                                 "soc_min", "soc_max") if k in ep}
            for ep in packet["recent_episodes"]
        ],
        "feedback_preference": packet["feedback"]["preference"],
        "reward_source": packet["reward_source"],
    }
    return (
        "You are the fuel efficiency reviewer for an FCEV Energy Management System reward function.\n"
        "Your ONLY responsibility: assess H2 fuel efficiency alignment.\n\n"
        "Focus on:\n"
        "- h2_stats: H2 consumption trend (mean, min, max across episodes)\n"
        "- recent_episodes: mean_h2_cost vs mean_soc_cost ratio, mean_fcs_cost vs mean_h2_cost ratio, mean_objective_cost trend\n"
        "- feedback_preference: preferred vs dispreferred h2-related behavior\n"
        "- reward_source: h2_cost structure, w_h2/w_soc coefficient balance, and w_fcs_soh coefficient\n\n"
        "Hard constraints you must enforce:\n"
        "- h2_cost MUST stay proportional to h2_fcs: the intended form is effective_w_h2 * h2_fcs\n"
        "  where effective_w_h2 is a FIXED constant (e.g., w_h2, w_h2 * 1.10, 15.0)\n"
        "- Do NOT use self.h2_batt or self.h2_equal inside reward computation; they are logging-only\n"
        "- Do NOT split hydrogen accounting into an extra top-level term\n"
        "- Keep exactly four top-level cost components: soc_cost, h2_cost, fcs_soh_cost, batt_soh_cost\n"
        "- objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost; reward = -objective_cost\n\n"
        "ALLOWED vs FORBIDDEN for h2_cost:\n"
        "  ALLOWED (pure proportional — any fixed coefficient is fine):\n"
        "    h2_cost = float(self.w_h2) * h2_fcs\n"
        "    w_h2_eff = w_h2 * 1.10; h2_cost = w_h2_eff * h2_fcs  (constant multiplier)\n"
        "    h2_cost = 15.0 * h2_fcs            (hardcoded coefficient)\n"
        "  FORBIDDEN (state-dependent / relative — these create relative rewards and can increase total H2):\n"
        "    EMA baseline: h2_ema = (1-α)*h2_ema + α*h2_fcs; eff_bonus = f(h2_ema - h2_fcs)\n"
        "    activity_gate = h2_fcs / (0.25 * ref)  (running-average gate)\n"
        "    Any eff_bonus, efficiency_signal, or supplementary term added ON TOP of h2_cost\n"
        "  To increase h2 optimization pressure raise the fixed coefficient directly — but ONLY when\n"
        "  h2_cost fraction is below target. If h2_cost is already >= 50% of objective and h2_100km has\n"
        "  stalled, do NOT raise w_h2; the correct lever is widening the SOC deadband to enable buffering.\n\n"
        "Battery buffer utilization diagnostic:\n"
        "- Compute per-episode SOC swing: soc_max - soc_min for each recent episode.\n"
        "- SOC swing 0.05–0.20 within [0.4, 0.8]: HEALTHY — battery is buffering FCS load.\n"
        "- SOC swing < 0.05 consistently: battery underutilized — FCS follows load directly\n"
        "  → idling on low-speed cycles → H2 waste. This is a fuel efficiency problem\n"
        "  even when in_bounds_rate is 100%. Classify as soft_revise; see levers in soft_revise below.\n"
        "  Do NOT increase w_h2 alone in this case — it will not address the root cause.\n"
        "- SOC out of [0.4, 0.8]: safety violation — defer to soc_safety reviewer.\n\n"
        "FCS operating quality signal (use mean_fcs_cost to diagnose operating point stress):\n"
        "- fcs_soh_cost = w_fcs_soh * dSOH_FCS accumulates degradation only from:\n"
        "    (a) low-load idling (P_fc < 0.05 * P_FC_max), (b) start-stop events, (c) rapid load changes\n"
        "  at efficient moderate-load steady operation dSOH_FCS is exactly zero\n"
        "- If mean_fcs_cost is consistently elevated across recent episodes:\n"
        "    FCS is frequently stressed at poor operating points (idling, on/off cycling, load spikes)\n"
        "    propose increase w_fcs_soh (10-30%) to strengthen the penalty on these events,\n"
        "    which incentivizes the agent to avoid low-load idling and prefer battery dispatch\n"
        "    WITHOUT requiring speed or load state information in the reward\n"
        "- w_fcs_soh tuning is a secondary lever; always assess battery buffer utilization first\n\n"
        "w_h2 / w_soc ratio check:\n"
        "- Compute approximate h2 fraction from recent_episodes: mean_h2_cost / mean_objective_cost\n"
        "- Target: h2_cost should be 30-50% of total mean_objective_cost\n"
        "- CEILING: if h2_cost is already >= 50% of objective AND h2_100km is stalled (not improving),\n"
        "  do NOT propose increasing w_h2 — the lever is reducing w_soc / widening the SOC deadband so the\n"
        "  battery can buffer FCS load; raising w_h2 further will not escape the local minimum\n"
        "- Aligned-gate minimum: h2_cost fraction must stay >= 25%; below that, the reward is underweighting fuel economy\n"
        "- If mean_h2_cost / mean_objective_cost < 0.25, w_h2 is likely underweighted → propose increase\n"
        "- If soc_cost has more than 3 terms (complex shaping), flag it as the likely root cause\n"
        "- If battery buffer utilization is low (SOC swing < 0.05), flag this as higher-priority root cause\n\n"
        "PASS graduation criteria (return 'pass' when ALL hold):\n"
        "- h2_cost is proportional to h2_fcs with a fixed coefficient (no EMA/gate/bonus added)\n"
        "- four-term objective skeleton is preserved\n"
        "- h2_cost contributes >= 25% of mean_objective_cost in recent episodes\n\n"
        "soft_revise criteria (only when pass criteria are NOT met):\n"
        "- h2_cost is structurally correct (pure proportional) but effective_w_h2 is too small (ONLY when h2_cost fraction < 30%)\n"
        "- OR h2_cost fraction is < 25% → propose increasing effective_w_h2 by 10-30% while keeping pure proportional form\n"
        "- OR mean_fcs_cost is consistently elevated → propose increase w_fcs_soh\n"
        "- OR SOC swing is consistently < 0.05 AND h2_100km is high → propose reduce w_soc or add deadband\n"
        "- Propose ONLY: reduce w_soc (if SOC swing is suppressed), increase effective_w_h2 by 10-30%,\n"
        "  increase w_fcs_soh by 10-30%, OR flag soc over-shaping as root cause\n"
        "- Do NOT propose adding new h2 shaping terms\n\n"
        "hard_reject criteria (use ONLY for genuine violations, not coefficient style choices):\n"
        "- reward directly rewards MORE H2 consumption (negative h2 cost)\n"
        "- h2_batt or h2_equal used inside reward computation logic (not logging)\n"
        "- four-term objective skeleton broken (h2 split into extra top-level terms)\n"
        "- h2_cost has a state-dependent component: EMA baseline, running average gate, "
        "  proximity gate, or any term where the h2_fcs multiplier changes based on episode history\n\n"
        "Output ONLY a compact JSON object inside ```json ... ``` with schema:\n"
        "{\n"
        '  "agent": "fuel_efficiency_reviewer",\n'
        '  "decision": "pass | soft_revise | hard_reject",\n'
        '  "risk_score": <0.0-1.0>,\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "h2_trend": "improving | stable | degrading | unknown",\n'
        '  "h2_cost_fraction": <float: estimated mean_h2_cost / mean_objective_cost from recent episodes>,\n'
        '  "fcs_quality_ratio": <float: estimated mean_fcs_cost / mean_h2_cost from recent episodes, null if unavailable>,\n'
        '  "soc_swing_mean": <float: mean of (soc_max - soc_min) across recent episodes, null if unavailable>,\n'
        '  "battery_buffer_utilized": <bool: true if soc_swing_mean >= 0.05>,\n'
        '  "summary": "<one sentence: h2 trend, battery buffer utilization, fcs operating quality, and decision reason>",\n'
        '  "proposals": [{"target": <str>, "direction": "increase_small|decrease_small|keep|remove", "reason": <str>}]\n'
        "}\n\n"
        "REVIEW_PACKET:\n"
        + json.dumps(focused, ensure_ascii=False, indent=2)
    )


def _hacking_prompt(packet: Dict[str, Any]) -> str:
    focused = {
        "iteration": packet["iteration"],
        "reward_source": packet["reward_source"],
        "note": (
            "h2_batt and h2_equal are LOGGING-ONLY variables in this EMS. "
            "Using them as primary reward signal (not just logging to self.info) is forbidden."
        ),
    }
    return (
        "You are the reward hacking reviewer for an FCEV Energy Management System reward function.\n"
        "Your ONLY responsibility: identify reward hacking risks, non-causal terms, or scale blow-up.\n\n"
        "Hard constraints — flag any violation:\n"
        "- Replay inputs: only SOC, P_batt, P_FCS, h2_fcs, dSOH_FCS, dSOH_batt are available.\n"
        "  Do NOT rely on step, episode_steps, progress, travel, remaining time, or terminal flags.\n"
        "- h2_batt and h2_equal are logging-only; do NOT use as reward inputs.\n"
        "- Exactly four top-level cost components; preserve objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost.\n"
        "- reward = -objective_cost; must remain a scalar float runnable in training and stub/smoke contexts.\n"
        "- All required logging keys must be present in self.info: "
        + ", ".join(REQUIRED_INFO_KEYS) + "\n\n"
        "Additional hacking vectors to check:\n"
        "- Non-causal terms (future or oracle information)\n"
        "- Trivially exploitable terms (e.g., staying stationary zeros out h2_fcs)\n"
        "- Scale explosion from large coefficients on unbounded inputs\n\n"
        "Over-shaping detection (flag as soft_revise if ANY of these are true):\n"
        "- soc_cost contains more than 3 distinct additive terms\n"
        "- h2_cost contains EMA-based bonuses, activity gates, proximity gates, or any relative-baseline term\n"
        "  beyond the base w_h2 * h2_fcs form\n"
        "- Total distinct named cost sub-terms across all four components exceeds 8\n"
        "- Any multiplicative gate on h2_cost keyed to SOC proximity that suppresses the h2 signal\n"
        "  (NOTE: a penalty-free deadband on soc_cost is NOT a gate on h2_cost — a wide soc_cost\n"
        "   deadband enables battery buffering and is ALLOWED at any width; do not flag it here)\n\n"
        "Decision rules:\n"
        "- hard_reject: non-causal term, replay-incompatible dependency, h2_batt/h2_equal as reward input,\n"
        "  or four-term skeleton / reward sign broken\n"
        "- soft_revise: over-shaping detected, scale blow-up risk, stub compatibility issue, or logging-contract violation\n"
        "- pass: no hard-reject conditions, no over-shaping, no exploitation vectors\n\n"
        "Output ONLY a compact JSON object inside ```json ... ``` with schema:\n"
        "{\n"
        '  "agent": "reward_hacking_reviewer",\n'
        '  "decision": "pass | soft_revise | hard_reject",\n'
        '  "risk_score": <0.0-1.0>,\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "summary": "<one sentence: key findings>",\n'
        '  "exploit_flags": [<str>],\n'
        '  "proposals": [{"target": <str>, "direction": "increase_small|decrease_small|keep|remove", "reason": <str>}]\n'
        "}\n\n"
        "REVIEW_PACKET:\n"
        + json.dumps(focused, ensure_ascii=False, indent=2)
    )


def _fc_durability_prompt(packet: Dict[str, Any]) -> str:
    """Priority 2: FC Durability specialist — fuel-efficiency-first persona."""
    focused = {
        "iteration": packet["iteration"],
        "h2_stats": packet["h2_stats"],
        "recent_episodes": [
            {k: ep[k] for k in ("episode", "mean_fcs_cost", "mean_h2_cost",
                                 "h2_100km", "mean_objective_cost") if k in ep}
            for ep in packet["recent_episodes"]
        ],
        "reward_source": packet["reward_source"],
        "diagnostic_hint": packet.get("diagnostic_result", {}).get(
            "reviewer_hints", {}).get("fc_durability", ""),
    }
    return (
        "You are the FC durability specialist for an FCEV Energy Management System reward function.\n"
        "CRITICAL PRIORITY: Fuel efficiency (h2_100km, g/km) is the PRIMARY objective.\n"
        "FCS durability is important ONLY when it directly and provably impacts fuel efficiency.\n"
        "Do NOT recommend w_fcs_soh changes that would increase hydrogen consumption.\n\n"
        "Your ONLY responsibility: assess FCS operating quality and durability cost signal.\n\n"
        "FCS operating modes and their fuel impact:\n"
        "- Efficient zone (P_fc = 40-80% P_FC_max): dSOH_FCS ≈ 0, low H2 per kW — DESIRED\n"
        "- Idling (P_fc < 5% P_FC_max): HIGH dSOH_FCS accumulation; poor efficiency → H2 waste\n"
        "- Start-stop events: moderate dSOH_FCS spike\n"
        "- Rapid load changes: brief dSOH_FCS spike, often unavoidable\n\n"
        "Focus on:\n"
        "- mean_fcs_cost / mean_h2_cost ratio (fcs_ratio):\n"
        "  fcs_ratio > 0.15 consistently → FCS idling is excessive → H2 waste\n"
        "  fcs_ratio < 0.05 → FCS degradation minimal; w_fcs_soh is adequately set\n"
        "- h2_100km trend: if high AND fcs_ratio is high → FCS idling is causing H2 waste\n"
        "  → RECOMMEND: increase w_fcs_soh to penalize idling/cycling\n"
        "- h2_100km trend: if low (good) regardless of fcs_ratio → FCS management already optimal\n"
        "  → return 'pass'; do NOT recommend changes\n\n"
        "Hard constraints:\n"
        "- fcs_soh_cost = w_fcs * dSOH_FCS: keep this form exactly\n"
        "- Do NOT recommend removing fcs_soh_cost (it penalizes exactly the degradation events)\n"
        "- Do NOT recommend more than 25% w_fcs_soh change per iteration\n"
        "- NEVER recommend changes that would increase h2_100km\n\n"
        "PASS graduation criteria (return 'pass' when ALL hold):\n"
        "- fcs_ratio <= 0.15 OR h2_100km is at or below the mean of the last 3 iterations\n"
        "- fcs_soh_cost form is correct (w_fcs * dSOH_FCS)\n\n"
        "soft_revise criteria:\n"
        "- fcs_ratio > 0.15 consistently AND h2_100km is high\n"
        "- Propose ONLY: increase w_fcs_soh by 10-25% to penalize FCS idling/cycling\n\n"
        "Output ONLY a compact JSON object inside ```json ... ``` with schema:\n"
        "{\n"
        '  "agent": "fc_durability_reviewer",\n'
        '  "decision": "pass | soft_revise | hard_reject",\n'
        '  "risk_score": <0.0-1.0>,\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "fcs_quality_ratio": <float: mean_fcs_cost / mean_h2_cost, null if unavailable>,\n'
        '  "fcs_idling_risk": "high | medium | low",\n'
        '  "fuel_efficiency_impact": "positive | neutral | negative",\n'
        '  "summary": "<one sentence: FCS operating quality and fuel efficiency impact>",\n'
        '  "proposals": [{"target": <str>, "direction": "increase_small|decrease_small|keep", "reason": <str>}]\n'
        "}\n\n"
        "REVIEW_PACKET:\n"
        + json.dumps(focused, ensure_ascii=False, indent=2)
    )


def _battery_usage_prompt(packet: Dict[str, Any]) -> str:
    """Priority 2: Battery Usage specialist — fuel-efficiency-first persona."""
    focused = {
        "iteration": packet["iteration"],
        "soc_bounds": packet["soc_bounds"],
        "h2_stats": packet["h2_stats"],
        "recent_episodes": [
            {k: ep[k] for k in ("episode", "soc_min", "soc_max",
                                 "in_bounds_rate_pct", "h2_100km",
                                 "mean_h2_cost", "mean_soc_cost") if k in ep}
            for ep in packet["recent_episodes"]
        ],
        "reward_source": packet["reward_source"],
        "diagnostic_hint": packet.get("diagnostic_result", {}).get(
            "reviewer_hints", {}).get("battery_usage", ""),
    }
    return (
        "You are the battery usage specialist for an FCEV Energy Management System reward function.\n"
        "CRITICAL PRIORITY: Fuel efficiency (h2_100km, g/km) is the PRIMARY objective.\n"
        "Battery analysis exists to ENABLE fuel efficiency via FCS load-leveling.\n"
        "SOC constraint enforcement is secondary to fuel efficiency improvement.\n\n"
        "Your ONLY responsibility: assess battery buffer utilization quality.\n\n"
        "Battery buffer principle:\n"
        "- The battery MUST actively buffer FCS load: absorb demand peaks, smooth FCS output\n"
        "- This requires SOC to SWING within [soc_low, soc_high]\n"
        "- SOC swing 0.05–0.25 within bounds: OPTIMAL — battery is load-leveling effectively\n"
        "- SOC swing < 0.05 consistently: battery NOT used → FCS follows load directly\n"
        "  → FCS idles on city cycles → H2 WASTE — this is a fuel efficiency EMERGENCY\n"
        "- SOC out of bounds: safety violation (defer to soc_safety reviewer)\n\n"
        "Focus on:\n"
        "- Per-episode SOC swing: soc_max - soc_min for each recent episode\n"
        "- mean SOC swing across recent episodes\n"
        "- h2_100km vs SOC swing correlation:\n"
        "  low swing + high h2_100km → battery underutilization is causing fuel inefficiency → ROOT CAUSE\n"
        "- in_bounds_rate_pct: if 100% AND swing < 0.05 → SOC is frozen near target (PROBLEM)\n\n"
        "Diagnosis rules:\n"
        "- mean_swing < 0.05 AND h2_100km above target:\n"
        "    Battery buffering SUPPRESSED → FCS follows load → H2 waste\n"
        "    Root cause: w_soc too tight OR soc_cost over-shaped\n"
        "    RECOMMEND: reduce w_soc 15-25% OR introduce deadband db≈0.03-0.06\n"
        "- mean_swing 0.05–0.25: healthy battery buffering → return 'pass'\n"
        "- mean_swing > 0.30 AND in_bounds_rate < 90%: excessive oscillation → flag to soc_safety\n\n"
        "FORBIDDEN recommendations:\n"
        "- Do NOT recommend increasing k_oob when swing is already small\n"
        "- Do NOT recommend tightening SOC tracking\n"
        "- Do NOT propose new SOC penalty layers\n"
        "- Only propose: reduce w_soc, add/widen deadband, or flag soc over-shaping\n\n"
        "PASS graduation criteria (return 'pass' when ALL hold):\n"
        "- mean_swing >= 0.05 in most recent episodes\n"
        "- in_bounds_rate_pct >= 90% in most recent episodes\n\n"
        "soft_revise criteria:\n"
        "- mean_swing < 0.05 (battery not buffering)\n"
        "- Propose ONLY: reduce w_soc by 15-25%, OR add/widen deadband in soc_cost\n\n"
        "Output ONLY a compact JSON object inside ```json ... ``` with schema:\n"
        "{\n"
        '  "agent": "battery_usage_reviewer",\n'
        '  "decision": "pass | soft_revise | hard_reject",\n'
        '  "risk_score": <0.0-1.0>,\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "mean_soc_swing": <float: mean of (soc_max - soc_min) across recent episodes, null if unavailable>,\n'
        '  "battery_buffering_active": <bool: true if mean_soc_swing >= 0.05>,\n'
        '  "fuel_efficiency_impact": "positive | neutral | negative",\n'
        '  "h2_suppression_risk": "high | medium | low",\n'
        '  "summary": "<one sentence: battery buffer state and fuel efficiency impact>",\n'
        '  "proposals": [{"target": <str>, "direction": "increase_small|decrease_small|keep|add_deadband", "reason": <str>}]\n'
        "}\n\n"
        "REVIEW_PACKET:\n"
        + json.dumps(focused, ensure_ascii=False, indent=2)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Blackboard MAS: observation scope + blackboard/conflict schema injection
# ─────────────────────────────────────────────────────────────────────────────

# Keyword sets for per-reviewer insight filtering (Condition 3 — observation scope).
# Each reviewer only sees memory insights whose text contains at least one of its keywords.
_REVIEWER_INSIGHT_KEYWORDS: Dict[str, List[str]] = {
    "soc_safety":      ["soc", "oob", "out_of_bounds", "in_bounds", "safety", "swing"],
    "fuel_efficiency": ["h2", "fuel", "efficiency", "battery", "buffering", "g/km", "100km"],
    "reward_hacking":  ["hack", "exploit", "shaping", "over_shaping", "reward_collapse"],
    "fc_durability":   ["fcs", "fc", "durability", "cycle", "idling"],
    "battery_usage":   ["battery", "swing", "buffering", "h2_suppression", "deadband"],
}

# Suffix appended to every reviewer prompt to inject blackboard and conflict fields (①②).
_REVIEWER_BLACKBOARD_SUFFIX = (
    "\n\nAdditionally, append these two optional fields to your JSON output:\n"
    '  "blackboard_post": "<1-2 sentences describing a recurring pattern you observe in your '
    'domain across iterations — use prior_insights below if available. null if nothing notable.>",\n'
    '  "conflict_signal": {"against": "<agent_name>", "issue": "<brief description of the conflict>"} '
    "or null — use this if you detect that your assessment directly conflicts with another agent's domain.\n"
    "These fields are read by the mediator and stored in shared iteration memory.\n"
)


def _filter_insights(
    insight_memory_list: List[Dict[str, Any]],
    keywords: List[str],
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    """Return the most recent insight entries whose text contains any of the keywords."""
    result = []
    for entry in reversed(insight_memory_list):
        text = json.dumps(entry, ensure_ascii=False).lower()
        if any(kw.lower() in text for kw in keywords):
            result.append(entry)
        if len(result) >= max_items:
            break
    return list(reversed(result))


_REVIEWER_PROMPT_BUILDERS = {
    "soc_safety":      _soc_prompt,
    "fuel_efficiency": _fuel_prompt,
    "reward_hacking":  _hacking_prompt,
    "fc_durability":   _fc_durability_prompt,
    "battery_usage":   _battery_usage_prompt,
}


# ─────────────────────────────────────────────────────────────────────────────
# Trainability reviewer: pure Python (no LLM)
# Mirrors reward_lr_tuner._training_trend_summary + auto_tune_lrs logic.
# Runs before the proposer, so reward_scale_ratio is not yet computable.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_trainability_from_packet(review_packet: Dict[str, Any]) -> Dict[str, Any]:
    """Compute trainability assessment and LR proposal from training_trend signals."""
    trend = review_packet.get("training_trend", {})
    best_episode_ratio = float(trend.get("best_episode_ratio", 1.0))
    tail_gap_ratio = float(trend.get("tail_gap_ratio", 0.0))
    trend_factor = float(trend.get("trend_factor", 1.0))
    early_best_threshold = 0.35
    tail_gap_threshold = 0.06

    lr_factor = max(0.55, min(1.10, trend_factor))

    base_warmup = float(
        review_packet.get("lr_settings", {}).get("lr_warmup_ratio", 0.10)
    )
    warmup_ratio = max(0.10, min(0.20, base_warmup + 0.10 * max(0.0, 1.0 - lr_factor)))

    if (
        trend_factor < 0.80
        or best_episode_ratio < early_best_threshold
        or tail_gap_ratio > tail_gap_threshold
    ):
        decision = "soft_revise"
        risk_trend = max(0.0, (0.80 - trend_factor) / 0.30)
        risk_early = max(0.0, (early_best_threshold - best_episode_ratio) / early_best_threshold)
        risk_tail = max(0.0, (tail_gap_ratio - tail_gap_threshold) / tail_gap_threshold)
        risk_score = round(min(1.0, 0.30 + 0.50 * max(risk_trend, risk_early, risk_tail)), 3)
    else:
        decision = "pass"
        risk_score = round(max(0.0, 0.10 * (1.0 - trend_factor)), 3)

    return {
        "agent": "trainability_scheduler_reviewer",
        "decision": decision,
        "risk_score": risk_score,
        "confidence": 1.0,
        "signals": {
            "best_episode_ratio": round(best_episode_ratio, 4),
            "tail_gap_ratio": round(tail_gap_ratio, 4),
            "reward_scale_ratio": 1.0,
        },
        "lr_proposal": {
            "lr_factor": round(lr_factor, 4),
            "critic_factor": round(lr_factor, 4),
            "actor_factor": round(lr_factor, 4),
            "warmup_ratio": round(warmup_ratio, 4),
            "alpha_change": "keep",
        },
        "_source": "python_computation",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reviewer runner
# ─────────────────────────────────────────────────────────────────────────────

# Agents that participate in peer-rebuttal debate rounds (Condition 1).
# reward_hacking, fc_durability, scale_reviewer, trainability_scheduler do not debate.
DEBATE_AGENTS = ["soc_safety", "battery_usage", "fuel_efficiency"]


def _build_rebuttal_prefix(
    reviewer_name: str,
    debate_round: int,
    debate_thread: List[Dict[str, Any]],
) -> str:
    """Build the peer-opinion preamble injected before the reviewer's own task prompt."""
    others = [m for m in debate_thread if m["sender"] != reviewer_name]
    if not others:
        return ""
    lines = [
        f"# Debate Round {debate_round}: Peer Opinions (consider and respond to these)",
        "CRITICAL: Fuel efficiency (h2_100km, g/km) is ALWAYS the primary objective. "
        "Maintain this priority in your updated assessment.",
        "",
    ]
    for msg in others:
        lines.append(f"## {msg['sender']} opinion (round {msg['round_idx']}):")
        lines.append(msg["content"])
        lines.append("")
    lines.append(
        "You may revise your assessment in light of the above peer opinions. "
        "Output the same JSON schema as before."
    )
    lines.append("\n# Your Assessment Task (same as before, apply peer context above):\n")
    return "\n".join(lines)


def _run_reviewer(
    client,
    reviewer_name: str,
    review_packet: Dict[str, Any],
    args,
    usage_records: List[Dict[str, Any]],
    usage_path: Path,
    model: Optional[str] = None,
    debate_round: int = 0,
    debate_thread: Optional[List[Dict[str, Any]]] = None,
    ctx: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """Call one reviewer LLM and parse its JSON output.

    model overrides client.model for this call (per-agent model selection).
    debate_round / debate_thread: when > 0, prepend peer opinions for rebuttal round.

    Injects three blackboard features into every call:
      ① blackboard_post field — domain observation for shared memory
      ② conflict_signal field — explicit inter-agent conflict flagging
      ③ relevant_insights — filtered memory slice for this reviewer's scope
    """
    base_prompt = _REVIEWER_PROMPT_BUILDERS[reviewer_name](review_packet)

    # ③ Observation scope: inject only domain-relevant insights from iteration memory
    keywords = _REVIEWER_INSIGHT_KEYWORDS.get(reviewer_name, [])
    all_insights = (
        review_packet.get("prior_iterations_summary", {}).get("insights", [])
    )
    relevant = _filter_insights(all_insights, keywords)
    scope_section = ""
    if relevant:
        scope_section = (
            "\n\nRELEVANT_MEMORY_INSIGHTS (filtered for your domain — "
            "use to detect recurring patterns across iterations):\n"
            + json.dumps(relevant, ensure_ascii=False, indent=2)
            + "\n"
        )

    # ①② Blackboard + conflict schema injection
    prompt = base_prompt + scope_section + _REVIEWER_BLACKBOARD_SUFFIX

    # Rebuttal prefix for debate rounds
    if debate_round > 0 and debate_thread:
        prompt = _build_rebuttal_prefix(reviewer_name, debate_round, debate_thread) + prompt

    stage_label = (
        f"reviewer:{reviewer_name}:round{debate_round}"
        if debate_round > 0
        else f"reviewer:{reviewer_name}"
    )
    max_tokens = max(800, int(getattr(args, "llm_max_tokens", 3500)) // 4)
    tools = get_tools_for_agent(reviewer_name) if ctx is not None else []
    tool_exec = (
        (lambda name, inp, _rn=reviewer_name: execute_tool(name, inp, ctx, agent_name=_rn))
        if ctx is not None else None
    )

    response = _tracked_complete_with_tools(
        client=client,
        usage_records=usage_records,
        usage_path=usage_path,
        stage=stage_label,
        prompt=prompt,
        tools=tools,
        tool_executor=tool_exec,
        temperature=float(getattr(args, "llm_temperature", 0.2)),
        max_tokens=max_tokens,
        timeout_sec=int(getattr(args, "llm_timeout_sec", 240)),
        model=model,
        ctx=ctx,
    )
    try:
        result = _extract_json(response)
    except (ValueError, json.JSONDecodeError) as e:
        result = {
            "agent": f"{reviewer_name}_reviewer",
            "decision": "soft_revise",
            "risk_score": 0.5,
            "confidence": 0.1,
            "_parse_error": str(e),
        }
    result["_raw_response"] = response
    # Normalise: ensure blackboard/conflict keys exist even if LLM omitted them
    result.setdefault("blackboard_post", None)
    result.setdefault("conflict_signal", None)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

# Appended to the orchestrator prompt when the counterfactual evaluator appeals.
_ORCHESTRATOR_APPEAL_SUFFIX = (
    "\n\n# APPEAL: Counterfactual Risk Evaluator (revise reward_spec now)\n"
    "The counterfactual risk evaluator found HIGH-RISK entries in must_change that the "
    "trade-off mediator did not block. You MUST revise must_change to remove or constrain "
    "these directives before the proposer acts on them:\n"
    "__APPEAL_GROUNDS__\n\n"
    "Fuel efficiency protection takes ABSOLUTE PRIORITY over SOC pressure increases. "
    "Output the same JSON schema as before, with the problematic must_change entries either "
    "removed or replaced with fuel-efficiency-safe alternatives."
)


_ORCHESTRATOR_PROMPT_TEMPLATE = (
    "You are the orchestrator for an FCEV reward refinement pipeline.\n\n"
    "You receive five reviewer outputs. Three are from LLM reviewers; "
    "scale_reviewer and trainability_scheduler_reviewer are pre-computed by deterministic Python "
    "(confidence=1.0, _source=python_computation) — treat them as authoritative.\n\n"
    "Historical reward guard (deterministic, must obey):\n"
    "- The current get_reward(self) has already been trained, so returning it unchanged is invalid.\n"
    "- Reverting to any previously trained reward variant is invalid.\n"
    "- If a reviewer proposal implicitly suggests rollback/no-op, rewrite reward_spec so the proposer must make a new local refinement instead.\n\n"
    "PRIMARY OBJECTIVE of this FCEV EMS:\n"
    "  Minimize hydrogen consumption (fuel economy) while maintaining charge-sustaining SOC.\n"
    "  SOC constraint is a MEANS to enable fuel economy, not an end in itself.\n"
    "  Do NOT allow soc_cost complexity to grow at the expense of the h2 optimization signal.\n\n"
    "Battery load-leveling principle:\n"
    "  SOC should swing within [0.4, 0.8] (swing 0.05–0.20 = healthy battery buffering).\n"
    "  SOC locked near target → FCS tracks load directly → idling on city cycles → H2 waste.\n"
    "  SOC outside [0.4, 0.8] → safety violation. Both failure modes must be distinguishable.\n\n"
    "SOC complexity budget (enforce before building reward_spec):\n"
    "- Count distinct additive terms in soc_cost from the current reward_source.\n"
    "- If soc_term_count >= 3 (or soc_safety reports over_shaping=true):\n"
    "    * Do NOT add new SOC penalty layers even if soc_safety returns soft_revise\n"
    "    * Instead, include 'simplify soc_cost to at most 3 terms' in must_change\n"
    "- soc_cost is allowed AT MOST: (1) base quadratic (pure or deadband form) + (2) OOB quadratic + (3) bounded recovery\n"
    "- Exponential barriers, margin zones, danger zones, and warn/caution/critical bands are FORBIDDEN\n\n"
    "CRITICAL: overall_decision MUST NEVER be 'reject'.\n"
    "  Reason: 'reject' causes the proposer to be skipped entirely and the current broken reward\n"
    "  to be reused unchanged in the next iteration — which is always worse than a repair attempt.\n"
    "  No matter how severe the reviewer flags, always output 'revise' so the proposer can fix it.\n\n"
    "Apply risk precedence:\n"
    "  1. soc_safety hard_reject → revise with urgent SOC repair (remove forbidden terms, fix structure)\n"
    "  2. reward_hacking hard_reject → revise with urgent anti-hacking repair\n"
    "  3. reward_hacking over-shaping flag → simplify before anything else\n"
    "  2.5. scale_reviewer hard_reject or soft_revise → MANDATORY: add every entry in\n"
    "     scale_reviewer.prescribed_corrections to must_change as\n"
    "     'reduce <coeff> to <= <prescribed_value> (scale cap violated)'\n"
    "     Scale caps are CEILINGS — they override any must_change from soc_safety or fuel_efficiency\n"
    "     that would push a coefficient above its cap. If soc_safety says 'increase k_oob' but\n"
    "     scale_reviewer caps k_oob at 3.75, the cap wins: use min(safety-required, prescribed).\n"
    "  4. fuel_efficiency (HIGH priority — h2 is the primary objective):\n"
    "     - battery_buffer_utilized=false AND h2 high → root cause is over-tight soc_cost; reduce w_soc or add deadband\n"
    "     - otherwise: tune w_h2/w_soc ratio and fcs_quality signal\n"
    "     - IMPORTANT: if scale_reviewer already flagged w_h2_multiplier as over-cap, do NOT increase it further\n"
    "  5. soc_safety soft_revise — address ONLY with coefficient tuning, never new SOC terms\n"
    "  6. trainability_scheduler — LR adjustment\n\n"
    "Decision rules (deterministic — apply in order, first match wins):\n"
    "- ANY hard_reject (from any reviewer) → overall_decision = 'revise'"
    " (urgent repair; proposer must fix the flagged violation)\n"
    "- soc_safety = pass AND fuel_efficiency = pass AND reward_hacking = pass AND scale_reviewer = pass"
    " AND fuel_efficiency.battery_buffer_utilized = true (or null)"
    " → overall_decision = 'accept'\n"
    "- soc_safety = pass AND fuel_efficiency = pass AND reward_hacking = pass AND scale_reviewer = pass"
    " AND fuel_efficiency.battery_buffer_utilized = false"
    " → overall_decision = 'revise' (battery buffering suppressed; reduce w_soc or add deadband)\n"
    "- soc_safety = pass AND reward_hacking = pass AND scale_reviewer = pass"
    " AND only fuel_efficiency = soft_revise"
    " → overall_decision = 'revise' (h2 tuning only)\n"
    "- Only trainability_scheduler has high risk (others pass)"
    " → overall_decision = 'accept_but_conservative'\n"
    "- Otherwise majority soft_revise → overall_decision = 'revise'\n\n"
    "When building reward_spec:\n"
    "STEP 1 — Assess SOC structure:\n"
    "  - soc_safety = pass AND soc_swing_ok = true: leave soc_cost untouched (SOC + buffering both healthy)\n"
    "  - soc_safety = pass BUT soc_swing_ok = false OR fuel_efficiency.battery_buffer_utilized = false:\n"
    "    → battery buffering suppressed; add ONE entry to must_change:\n"
    "      'reduce w_soc by 10-20% OR introduce deadband db≈0.03–0.05 in soc_cost base term'\n"
    "    → also add 'increase w_fcs_soh by 10-20% to penalize FCS idling'\n"
    "    (Both signals point to the same root cause — only one must_change entry needed.)\n"
    "  - soc_safety = soft_revise (OOB too high): add 'increase k_oob coefficient' to must_change only\n\n"
    "STEP 2 — Check reward_hacking over_shaping flag:\n"
    "  - If over-shaping detected: add 'remove excessive SOC sub-terms, keep at most 3' to must_change\n"
    "    and 'remove EMA/proximity-gated terms if present' to must_change\n\n"
    "STEP 3 — H2 optimization from fuel_efficiency:\n"
    "  - If battery buffering was already flagged in STEP 1: do NOT increase w_h2 alone\n"
    "  - Otherwise: increase w_h2 if h2_cost fraction < 25%, or increase w_fcs_soh if fcs_quality elevated\n"
    "  - Forbidden: adding eff_bonus, h2_ema, activity_gate, proximity_gate on top of h2_cost\n\n"
    "STEP 3.5 — Coefficient scale control from scale_reviewer:\n"
    "  - If scale_reviewer.decision != 'pass':\n"
    "    → For each entry in scale_reviewer.prescribed_corrections, add to must_change:\n"
    "      {'target': '<coeff>', 'direction': 'reduce to <= <prescribed_value>'}\n"
    "    → These directives take precedence over any STEP 1/3 directive that would increase\n"
    "      the same coefficient beyond its prescribed cap.\n"
    "  - If scale_reviewer.decision = 'pass': no scale action needed.\n\n"
    "must_keep must include:\n"
    "- 'soc_cost base quadratic target-tracking term: pure form w_soc*(soc-soc_target)^2 OR deadband form w_soc*k*max(0,|soc-soc_target|-db)^2'\n"
    "- 'four-term additive objective skeleton: soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost'\n"
    "- 'h2_cost = w_h2 * h2_fcs (pure proportional, no relative-baseline bonus)'\n\n"
    "forbidden must always include:\n"
    "- 'Adding new in-bounds SOC penalty layers (exponential barrier, margin zone, danger zone, warn/caution/critical bands)'\n"
    "- 'EMA-based or relative-baseline h2 efficiency bonus terms on top of h2_cost'\n"
    "- 'SOC proximity gates that suppress h2_cost signal'\n"
    "- 'Removing k_oob OOB term when using deadband soc_cost form'\n"
    "- 'Coefficient scale ceiling violations: k_oob > 8.0, h2 effective multiplier > 4.0, soc effective multiplier > 2.5, k_inbounds > 5.0'\n\n"
    "For lr_spec: copy lr_factor and warmup_ratio directly from "
    "trainability_scheduler_reviewer.lr_proposal (authoritative, do not override).\n\n"
    "Output ONLY a compact JSON object inside ```json ... ``` with schema:\n"
    "{\n"
    '  "overall_decision": "accept | accept_but_conservative | revise | reject",\n'
    '  "blocking_issues": [<str>],\n'
    '  "reward_spec": {\n'
    '    "must_keep": [<str>],\n'
    '    "must_change": [{"target": <str>, "direction": <str>}],\n'
    '    "forbidden": [<str>]\n'
    "  },\n"
    '  "lr_spec": {"lr_factor": <float>, "warmup_ratio": <float>},\n'
    '  "rationale": "<2-3 sentences max>"\n'
    "}\n\n"
    "HISTORICAL_REWARD_GUARD:\n"
    "__HISTORICAL_GUARD__\n\n"
    "REVIEWER_OUTPUTS:\n"
    "__REVIEWER_OUTPUTS__"
)


def _fallback_orchestrator_spec(reviewer_outputs: Dict[str, Dict[str, Any]], err: str) -> Dict[str, Any]:
    """Deterministic fallback when orchestrator LLM fails to produce valid JSON."""
    hard_rejecters = [
        name for name in ("soc_safety", "reward_hacking", "scale_reviewer")
        if reviewer_outputs.get(name, {}).get("decision") == "hard_reject"
    ]
    if hard_rejecters:
        decision = "reject"
    else:
        soft_revisers = [
            name for name, out in reviewer_outputs.items()
            if out.get("decision") == "soft_revise"
        ]
        decision = "revise" if soft_revisers else "accept"

    trainability = reviewer_outputs.get("trainability_scheduler", {})
    lr_proposal = trainability.get("lr_proposal", {})

    return {
        "overall_decision": decision,
        "blocking_issues": [f"hard_reject:{r}" for r in hard_rejecters],
        "reward_spec": {"must_keep": [], "must_change": [], "forbidden": []},
        "lr_spec": {
            "lr_factor": float(lr_proposal.get("lr_factor", 1.0)),
            "warmup_ratio": float(lr_proposal.get("warmup_ratio", 0.10)),
        },
        "rationale": f"fallback (orchestrator parse error: {err})",
        "_fallback": True,
    }


def _run_orchestrator(
    client,
    reviewer_outputs: Dict[str, Dict[str, Any]],
    historical_guard: Dict[str, Any],
    args,
    usage_records: List[Dict[str, Any]],
    usage_path: Path,
    model: Optional[str] = None,
    appeal_context: Optional[str] = None,
    ctx: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """Combine all reviewer outputs into a single orchestrator_spec.

    model overrides client.model for this call.
    appeal_context: if provided, appended to the prompt for a counterfactual-appeal re-run.
    """
    clean = {
        name: {k: v for k, v in out.items() if not k.startswith("_")}
        for name, out in reviewer_outputs.items()
    }
    prompt = (
        _ORCHESTRATOR_PROMPT_TEMPLATE
        .replace("__HISTORICAL_GUARD__", _render_historical_reward_guard(historical_guard))
        .replace("__REVIEWER_OUTPUTS__", json.dumps(clean, ensure_ascii=False, indent=2))
    )
    if appeal_context:
        prompt += appeal_context
    stage_name = "orchestrator:appeal" if appeal_context else "orchestrator"
    max_tokens = max(800, int(getattr(args, "llm_max_tokens", 3500)) // 3)
    orch_tools = get_tools_for_agent("orchestrator") if ctx is not None else []
    orch_exec = (
        (lambda name, inp: execute_tool(name, inp, ctx, agent_name="orchestrator"))
        if ctx is not None else None
    )

    response = _tracked_complete_with_tools(
        client=client,
        usage_records=usage_records,
        usage_path=usage_path,
        stage=stage_name,
        prompt=prompt,
        tools=orch_tools,
        tool_executor=orch_exec,
        temperature=float(getattr(args, "llm_temperature", 0.2)),
        max_tokens=max_tokens,
        timeout_sec=int(getattr(args, "llm_timeout_sec", 240)),
        model=model,
        ctx=ctx,
    )
    try:
        spec = _extract_json(response)
    except (ValueError, json.JSONDecodeError) as e:
        spec = _fallback_orchestrator_spec(reviewer_outputs, str(e))

    spec["_raw_response"] = response
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# Priority 3: Trade-off Mediator
# Resolves conflicts between specialist reviewers.
# CRITICAL: fuel efficiency (h2_100km, g/km) ALWAYS takes priority over SOC tightening.
# ─────────────────────────────────────────────────────────────────────────────

_TRADEOFF_MEDIATOR_PROMPT_TEMPLATE = (
    "You are the trade-off mediator for an FCEV reward refinement multi-agent pipeline.\n\n"
    "ABSOLUTE PRIORITY HIERARCHY (non-negotiable):\n"
    "  1. FUEL EFFICIENCY (h2_100km, g/km reduction) — PRIMARY OBJECTIVE\n"
    "  2. SOC safety (OOB prevention) — CONSTRAINT to enable fuel efficiency, NOT a goal\n"
    "  3. FCS durability — secondary constraint; never at expense of fuel efficiency\n"
    "  4. Training stability — tertiary\n\n"
    "Your job: identify conflicts between specialist reviewers and resolve them with the rules below.\n\n"
    "CONFLICT RESOLUTION RULES:\n"
    "Rule 1 — SOC pressure vs fuel efficiency:\n"
    "  If soc_safety says 'increase SOC pressure' but fuel_efficiency OR battery_usage says 'reduce SOC pressure':\n"
    "  → Check diagnostic.oob_rate:\n"
    "    oob_rate > 0.05 (>5%): allow MINIMUM k_oob increase needed for safety (cap at soft limit)\n"
    "    oob_rate <= 0.05 AND battery_usage.battery_buffering_active = false:\n"
    "      BLOCK all SOC pressure increases; MANDATE w_soc reduction or deadband instead\n"
    "    oob_rate <= 0.05 AND battery_usage.battery_buffering_active = true:\n"
    "      Current SOC setup is adequate; fuel_efficiency reviewer's proposals take priority\n\n"
    "Rule 2 — New SOC penalty layers:\n"
    "  Any proposal to add new SOC penalty terms/layers beyond the allowed 3-term budget:\n"
    "  → BLOCK unless soc_safety.over_shaping = false AND oob_rate > 0.10\n"
    "  → ALWAYS prefer coefficient tuning (k_oob, w_soc) over new penalty terms\n\n"
    "Rule 3 — FC durability vs h2 fraction:\n"
    "  If fc_durability says 'increase w_fcs_soh' AND fuel_efficiency says 'increase w_h2':\n"
    "  → h2_cost_fraction < 0.25: w_h2 increase takes PRIORITY; defer w_fcs_soh to next iter\n"
    "  → h2_cost_fraction >= 0.25 AND fcs_ratio > 0.15: allow both\n\n"
    "Rule 4 — H2 suppression emergency:\n"
    "  If battery_usage.h2_suppression_risk = 'high' OR diagnostic.failure_mode = 'h2_suppression':\n"
    "  → HIGHEST priority: MANDATE w_soc reduction or deadband BEFORE any other changes\n"
    "  → All SOC-tightening proposals are BLOCKED until this is resolved\n\n"
    "Output ONLY a compact JSON object inside ```json ... ``` with schema:\n"
    "{\n"
    '  "agent": "tradeoff_mediator",\n'
    '  "conflict_detected": <bool>,\n'
    '  "h2_suppression_risk_active": <bool: true if h2_suppression or battery_buffering_active=false>,\n'
    '  "mediated_priority": "fuel_efficiency | soc_safety | balanced",\n'
    '  "confirmed_changes": [{"target": <str>, "direction": <str>, "endorsed_by": [<str>]}],\n'
    '  "blocked_changes": [{"target": <str>, "reason": <str>, "blocked_reviewer": <str>}],\n'
    '  "fuel_priority_overrides": [{"context": <str>, "overridden": <str>, "reason": <str>}],\n'
    '  "rationale": "<2-3 sentences explaining mediator resolution logic>"\n'
    "}\n\n"
    "DIAGNOSTIC_RESULT:\n"
    "__DIAGNOSTIC__\n\n"
    "__DEBATE_TRAJECTORY__"
    "REVIEWER_OUTPUTS (final round):\n"
    "__REVIEWER_OUTPUTS__"
)


def _build_debate_trajectory_section(
    debate_thread: List[Dict[str, Any]],
) -> str:
    """Summarise per-agent position changes across debate rounds for the mediator.

    Returns an empty string when there is only one round (no rebuttal occurred).
    """
    if not debate_thread:
        return ""

    # Group messages by sender, sorted by round_idx
    from collections import defaultdict
    by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for msg in debate_thread:
        by_agent[msg["sender"]].append(msg)
    for msgs in by_agent.values():
        msgs.sort(key=lambda m: m["round_idx"])

    max_round = max(m["round_idx"] for m in debate_thread)
    if max_round == 0:
        return ""  # single round — nothing to show

    lines = [
        f"DEBATE_TRAJECTORY ({max_round + 1} rounds — how positions shifted during peer rebuttal):",
        "Use this to calibrate conflict resolution: an agent that already moved significantly",
        "between rounds has partially yielded — avoid forcing a larger concession.",
        "",
    ]
    for agent_name, msgs in sorted(by_agent.items()):
        lines.append(f"{agent_name}:")
        prev_decision = None
        for msg in msgs:
            try:
                payload = json.loads(msg["content"])
            except (json.JSONDecodeError, TypeError):
                payload = {}
            decision = payload.get("decision", "?")
            risk = payload.get("risk_score", "?")
            shift = ""
            if prev_decision is not None and decision != prev_decision:
                shift = f"  ← SHIFTED from {prev_decision}"
            lines.append(f"  round {msg['round_idx']}: decision={decision}, risk_score={risk}{shift}")
            prev_decision = decision
        lines.append("")

    lines.append("")
    return "\n".join(lines) + "\n"


def _run_tradeoff_mediator(
    client,
    reviewer_outputs: Dict[str, Dict[str, Any]],
    diagnostic_result: Dict[str, Any],
    args,
    usage_records: List[Dict[str, Any]],
    usage_path: Path,
    model: Optional[str] = None,
    debate_thread: Optional[List[Dict[str, Any]]] = None,
    appeal: Optional[Dict[str, Any]] = None,
    conflict_signals: Optional[List[Dict[str, Any]]] = None,
    ctx: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    clean_reviewers = {
        name: {k: v for k, v in out.items() if not k.startswith("_")}
        for name, out in reviewer_outputs.items()
    }
    clean_diag = {k: v for k, v in diagnostic_result.items() if not k.startswith("_")}
    trajectory_section = _build_debate_trajectory_section(debate_thread or [])
    prompt = (
        _TRADEOFF_MEDIATOR_PROMPT_TEMPLATE
        .replace("__DIAGNOSTIC__", json.dumps(clean_diag, ensure_ascii=False, indent=2))
        .replace("__DEBATE_TRAJECTORY__", trajectory_section)
        .replace("__REVIEWER_OUTPUTS__", json.dumps(clean_reviewers, ensure_ascii=False, indent=2))
    )
    if conflict_signals:
        signals_text = json.dumps(conflict_signals, ensure_ascii=False, indent=2)
        prompt += (
            "\n\nCONFLICT_SIGNALS (posted by reviewers — address these explicitly "
            "in blocked_changes or fuel_priority_overrides):\n"
            + signals_text
        )
    if appeal:
        prompt += (
            "\n\n# APPEAL FROM COUNTERFACTUAL EVALUATOR\n"
            "The following HIGH-RISK changes were not blocked in your previous decision. "
            "Please reconsider and explicitly block them if fuel efficiency protection requires it:\n"
            + appeal.get("appeal_grounds", "")
            + "\nOutput the same JSON schema with updated blocked_changes."
        )
    max_tokens = max(600, int(getattr(args, "llm_max_tokens", 3500)) // 4)
    med_tools = get_tools_for_agent("tradeoff_mediator") if ctx is not None else []
    med_exec = (
        (lambda name, inp: execute_tool(name, inp, ctx, agent_name="tradeoff_mediator"))
        if ctx is not None else None
    )
    response = _tracked_complete_with_tools(
        client=client,
        usage_records=usage_records,
        usage_path=usage_path,
        stage="tradeoff_mediator",
        prompt=prompt,
        tools=med_tools,
        tool_executor=med_exec,
        temperature=float(getattr(args, "llm_temperature", 0.2)),
        max_tokens=max_tokens,
        timeout_sec=int(getattr(args, "llm_timeout_sec", 240)),
        model=model,
        ctx=ctx,
    )
    try:
        result = _extract_json(response)
    except (ValueError, json.JSONDecodeError) as e:
        result = {
            "agent": "tradeoff_mediator",
            "conflict_detected": False,
            "h2_suppression_risk_active": False,
            "mediated_priority": "fuel_efficiency",
            "confirmed_changes": [],
            "blocked_changes": [],
            "fuel_priority_overrides": [],
            "rationale": f"mediator fallback (parse error: {e})",
            "_fallback": True,
        }
    result["_raw_response"] = response
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Priority 4: Counterfactual Risk Checker (pure Python)
# Evaluates orchestrator_spec.must_change for high-risk changes before proposer.
# ─────────────────────────────────────────────────────────────────────────────

def _check_counterfactual_risks(
    orchestrator_spec: Dict[str, Any],
    diagnostic_result: Dict[str, Any],
    reviewer_outputs: Dict[str, Dict[str, Any]],
    mediation_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Python-based counterfactual risk assessment; generates warning lines for proposer prompt."""
    must_change = orchestrator_spec.get("reward_spec", {}).get("must_change", [])
    oob_rate = float(diagnostic_result.get("oob_rate") or 0.0)
    mean_swing = diagnostic_result.get("mean_soc_swing")
    failure_mode = str(diagnostic_result.get("failure_mode") or "unknown")

    fuel_rev = reviewer_outputs.get("fuel_efficiency", {})
    battery_rev = reviewer_outputs.get("battery_usage", {})
    soc_rev = reviewer_outputs.get("soc_safety", {})

    h2_fraction = float(fuel_rev.get("h2_cost_fraction") or 0.5)
    battery_buffering = bool(battery_rev.get("battery_buffering_active", True))
    soc_over_shaping = bool(soc_rev.get("over_shaping", False))

    blocked_by_mediator = {
        str(b.get("target", "")).lower()
        for b in mediation_result.get("blocked_changes", [])
        if isinstance(b, dict)
    }
    h2_suppression_active = (
        bool(mediation_result.get("h2_suppression_risk_active", False))
        or failure_mode == "h2_suppression"
        or (mean_swing is not None and mean_swing < 0.05 and oob_rate < 0.05)
    )

    risk_flags: List[Dict[str, Any]] = []
    warning_lines: List[str] = []

    for change in must_change:
        if not isinstance(change, dict):
            continue
        target = str(change.get("target") or "")
        direction = str(change.get("direction") or "").lower()

        # Check if mediator already blocked this target
        if any(bt in target.lower() for bt in blocked_by_mediator):
            risk_flags.append({
                "change": f"{target}: {direction}",
                "risk": "high",
                "reason": f"trade-off mediator blocked '{target}' to protect fuel efficiency",
                "recommendation": "block",
            })
            warning_lines.append(
                f"COUNTERFACTUAL WARNING (high): '{target}' blocked by trade-off mediator "
                "(fuel efficiency priority). Do NOT implement this change."
            )
            continue

        # Rule 1: Increasing SOC pressure when battery buffering is suppressed
        soc_pressure_terms = ("k_oob", "w_soc", "k_inbounds", "soc_scale")
        if any(t in target.lower() for t in soc_pressure_terms) and "increase" in direction:
            if h2_suppression_active:
                risk_flags.append({
                    "change": f"{target}: {direction}",
                    "risk": "high",
                    "reason": (
                        f"h2_suppression_active=True: increasing SOC pressure worsens battery "
                        f"suppression (OOB={oob_rate:.1%}, swing={mean_swing})"
                    ),
                    "recommendation": "block_unless_oob_critical: only allow if OOB > 10%",
                })
                warning_lines.append(
                    f"COUNTERFACTUAL WARNING (high): '{target}' increase is HIGH RISK — "
                    f"battery buffering is already suppressed (mean_swing={mean_swing}). "
                    "Increasing SOC pressure will WORSEN fuel efficiency. "
                    f"Only allow if out-of-bounds rate > 10% (current: {oob_rate:.1%})."
                )

        # Rule 2: Reducing h2 weight when fraction already low
        if "w_h2" in target.lower() and any(d in direction for d in ("decrease", "reduce")):
            if h2_fraction < 0.30:
                risk_flags.append({
                    "change": f"{target}: {direction}",
                    "risk": "medium",
                    "reason": (
                        f"h2_cost_fraction={h2_fraction:.2f} already below 30%; "
                        "reducing w_h2 further underweights fuel economy"
                    ),
                    "recommendation": "floor: keep w_h2 >= current value",
                })
                warning_lines.append(
                    f"COUNTERFACTUAL WARNING (medium): '{target}' reduction flagged — "
                    f"h2_fraction={h2_fraction:.2f} already below 30%. "
                    "Keep w_h2 at or above its current value."
                )

        # Rule 3: Adding new SOC terms when already over-shaped
        if soc_over_shaping and "add" in direction and "soc" in target.lower():
            risk_flags.append({
                "change": f"{target}: {direction}",
                "risk": "medium",
                "reason": "soc_over_shaping=True: adding more SOC terms suppresses h2 gradient",
                "recommendation": "block: simplify soc_cost first",
            })
            warning_lines.append(
                f"COUNTERFACTUAL WARNING (medium): SOC term addition '{target}' blocked "
                "— soc_cost is already over-shaped. Simplify existing terms first."
            )

    overall_risk = (
        "high" if any(f["risk"] == "high" for f in risk_flags)
        else ("medium" if risk_flags else "low")
    )

    # appeal_flag: True when there are high-risk flags NOT already blocked by the mediator.
    # Blocked-by-mediator flags mean the mediator already caught the issue; no appeal needed.
    appeal_grounds_list = [
        f"{f['change']}: {f['reason']}"
        for f in risk_flags
        if f["risk"] == "high"
        and "blocked by trade-off mediator" not in f.get("reason", "")
    ]
    appeal_flag = bool(appeal_grounds_list)
    appeal_grounds = "; ".join(appeal_grounds_list)

    return {
        "agent": "counterfactual_evaluator",
        "risk_flags": risk_flags,
        "overall_risk": overall_risk,
        "must_change_safe": overall_risk == "low",
        "warning_lines": warning_lines,
        "appeal_flag": appeal_flag,
        "appeal_grounds": appeal_grounds,
        "risk_summary": (
            f"{len(risk_flags)} counterfactual risk flag(s). Overall: {overall_risk}. "
            + (f"Primary: {risk_flags[0]['reason']}" if risk_flags else "No high-risk changes detected.")
        ),
        "_source": "python_computation",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Proposer
# ─────────────────────────────────────────────────────────────────────────────

def _build_proposer_prompt(
    orchestrator_spec: Dict[str, Any],
    current_method: str,
    historical_guard: Dict[str, Any],
) -> str:
    clean_spec = {k: v for k, v in orchestrator_spec.items() if not k.startswith("_")}
    return (
        "You are the reward code proposer for an FCEV Energy Management System.\n\n"
        "PRIMARY OBJECTIVE: minimize hydrogen consumption (fuel economy) while maintaining "
        "charge-sustaining SOC. SOC is a constraint, not the goal.\n"
        "Charge-sustaining means: end-of-cycle SOC ≈ start SOC over a full driving cycle.\n"
        "It does NOT mean keeping SOC constant at target every step.\n\n"
        "Implement the orchestrator_spec by modifying current get_reward(self).\n\n"
        "Mandatory decision order (apply in this sequence):\n"
        "  1. Numerical safety: ensure all terms are finite, non-negative where required\n"
        "  2. Repair TPE preferred/dispreferred ordering if inverted\n"
        "  3. Improve h2 optimization: reduce hydrogen consumption signal\n"
        "  4. Maintain charge-sustaining SOC — allow battery to buffer FCS load (see Battery buffering below)\n"
        "  5. Preserve a simple, causal, learnable reward\n\n"
        "Battery buffering (apply when must_change specifies deadband or w_soc reduction):\n"
        "  SOC should swing within [0.4, 0.8]; swing 0.05–0.20 is healthy. SOC locked near target = no buffering.\n"
        "  Levers (in priority order):\n"
        "    (a) Deadband: soc_cost_base = w_soc * k_inbounds * max(0, |soc - soc_target| - db)^2\n"
        "        Zero penalty inside ±db → SOC free to swing. Always pair with k_oob OOB term.\n"
        "    (b) Reduce w_soc scaling factor to relax per-step tracking pressure\n"
        "    (c) Increase w_fcs_soh to penalize FCS idling/cycling\n"
        "  Do NOT increase w_h2 alone when battery buffering is the root cause.\n\n"
        "SOC dominance control:\n"
        "  If aligned-gate feedback or recent metrics indicate soc_cost/objective_cost is above 0.85, "
        "reduce SOC pressure before making any other SOC change.\n"
        "  Prefer reducing w_soc or k_inbounds, or using a small deadband, when SOC is already in bounds.\n"
        "  Do NOT increase k_oob or w_soc when OOB rate is <=5% and recent in_bounds_rate is >=95%.\n"
        "  Keep h2_cost a meaningful objective share; do not let SOC become the main optimization target.\n\n"
        "SOC simplicity hard constraint (NON-NEGOTIABLE):\n"
        "- soc_cost is allowed AT MOST 3 distinct additive terms:\n"
        "    (1) base quadratic — two allowed forms:\n"
        "        (a) Pure quadratic:    w_soc * (soc - soc_target)^2\n"
        "        (b) Deadband quadratic: w_soc * k_inbounds * max(0, |soc - soc_target| - db)^2\n"
        "            where db ∈ [0.02, 0.08] — penalty-free zone wide enough for GENUINE battery\n"
        "            buffering. Must be large enough that the healthy SOC swing (0.05–0.20) lives\n"
        "            mostly inside the free zone; a deadband < 0.02 re-penalizes buffering and is too narrow\n"
        "    (2) OOB quadratic strengthening: w_soc * k_oob * (dist_low^2 + dist_high^2)  [optional]\n"
        "        REQUIRED when using deadband form to keep hard protection at soc_bounds [0.4, 0.8]\n"
        "    (3) bounded recovery shaping  [only if OOB > 2% in recent episodes]\n"
        "- FORBIDDEN SOC terms: exponential barrier, margin zone, danger zone, warn band, "
        "  caution band, critical band, proximity penalty, any in-bounds penalty layer beyond base quadratic\n"
        "- If the current reward has more than 3 SOC terms, REMOVE the excess terms\n"
        "- When adding deadband: always pair with k_oob OOB term to prevent SOC leaving [0.4, 0.8]\n\n"
        "H2 cost hard constraint (NON-NEGOTIABLE):\n"
        "- h2_cost MUST be: float(self.w_h2) * h2_fcs  (pure proportional, per-step)\n"
        "- Do NOT add: EMA baseline, efficiency bonus, activity gate, proximity gate, "
        "  relative-deviation signal, tanh saturation, or any supplementary shaping term\n"
        "- h2_fcs is already a dense signal with a clear gradient — adding complexity hurts, not helps\n"
        "- To emphasize h2 optimization increase w_h2 directly ONLY when h2_cost fraction is below target\n"
        "- Target: h2_cost should be approximately 30-50% of total mean_objective_cost\n"
        "  CEILING: if h2_cost is already >= 50% of objective and h2_100km has stalled, do NOT raise w_h2;\n"
        "  the correct lever is widening the SOC deadband (db) to enable battery buffering\n\n"
        "Coefficient scale hard constraints (ABSOLUTE MAXIMUM — scale_reviewer enforces these):\n"
        "- k_oob: MAXIMUM 8.0. If the current value already exceeds 8.0, you MUST reduce it first.\n"
        "  If must_change says 'increase k_oob', the new value must still be <= 8.0.\n"
        "- h2 effective multiplier (h2_scale variable, or leading numeric in w_h2_eff assignment):\n"
        "  MAXIMUM 4.0. If current exceeds 4.0, reduce it REGARDLESS of h2_fraction target.\n"
        "- soc effective multiplier (leading numeric in w_soc_eff assignment or soc_scale variable):\n"
        "  MAXIMUM 2.5.\n"
        "- k_inbounds: MAXIMUM 5.0.\n"
        "- Per-iteration change rate: no coefficient may INCREASE by more than 100% vs. current value.\n"
        "  Example: if current h2_scale = 1.0, new h2_scale must be <= 2.0.\n"
        "  Example: if current k_oob = 3.0, new k_oob must be <= 6.0.\n"
        "If scale_reviewer.prescribed_corrections is non-empty, those values are MANDATORY —\n"
        "they override any must_change directive that would keep coefficients above the prescribed cap.\n\n"
        "Allowed modification scope:\n"
        "- Tune w_soc, k_inbounds, k_oob, db (deadband), w_h2, w_fcs_soh, w_batt_soh coefficients\n"
        "- Replace pure quadratic soc_cost base with deadband form (or vice versa) if orchestrator_spec requires it\n"
        "- Keep top-level objective decomposition unchanged\n"
        "- Do NOT use progress-aware, terminal-aware, or step-index shaping\n\n"
        "Output rules:\n"
        "- Output a single Python ```python ... ``` fenced block containing ONLY def get_reward(self):\n"
        "- Do NOT change the method signature\n"
        "- Do NOT use h2_batt or h2_equal as primary reward signals (logging-only)\n"
        "- Keep ALL logging keys in self.info: "
        + ", ".join(REQUIRED_INFO_KEYS) + "\n"
        "- Follow must_keep, must_change, and forbidden from the spec exactly\n"
        "- Do NOT introduce non-causal terms\n"
        "- Do NOT return the current reward unchanged\n"
        "- Do NOT revert to any previously trained reward variant\n"
        "- Exact structural matches to already-trained variants are rejected before TPE evaluation\n\n"
        "HISTORICAL_REWARD_GUARD:\n"
        + _render_historical_reward_guard(historical_guard)
        + "\n\n"
        "ORCHESTRATOR_SPEC:\n"
        + json.dumps(clean_spec, ensure_ascii=False, indent=2)
        + "\n\nCURRENT get_reward(self):\n"
        "```python\n"
        + current_method
        + "\n```\n\n"
        "After the code block, write a brief one-paragraph rationale explaining:\n"
        "- What changed and why\n"
        "- How the w_h2/w_soc balance was adjusted\n"
        "- Whether deadband was introduced or w_soc reduced, and expected effect on SOC swing\n"
        "- Final soc_cost term count"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aligned gate feedback builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_aligned_gate_feedback(aligned) -> str:
    """Build proposer retry prompt section from a failed AlignedGateResult."""
    lines = ["\n\n# Alignment Gate Feedback (must address — both layers required)"]
    rc = aligned.tac
    if not rc.passed and not rc.skipped:
        lines.append(f"- Layer 1 (TAC) FAILED: {rc.message}")
        lines.append(
            "  Revise get_reward so that episodes where the agent actually used "
            "less H2 (lower h2_100km) and maintained SOC better (higher in_bounds_rate) "
            "receive higher cumulative reward. The reward must reflect real physical "
            "efficiency, not just track objective_cost."
        )
    ct = aligned.constitution
    if not ct.passed and not ct.skipped:
        lines.append(f"- Layer 2 (Constitution) FAILED: {ct.message}")
        if not ct.h2_monotonicity_ok:
            lines.append(
                "  * H2 monotonicity violated: reward must strictly decrease "
                "when h2_fcs increases, all else equal."
            )
        if not ct.soc_boundary_ok:
            lines.append(
                "  * SOC boundary violated: reward must be lower when SOC is "
                "outside [soc_low, soc_high] than when SOC equals soc_target."
            )
        if not ct.dominance_ok:
            lines.append(
                "  * Dominance violated: one cost term exceeds 95 % of "
                "objective_cost on average, SOC dominates the objective, "
                "or H2 contribution is too small. Rebalance the weights."
            )
        if not getattr(ct, "soc_dominance_ok", True):
            lines.append(
                "  * SOC dominance violated: reduce w_soc/k_oob pressure "
                "unless recent out-of-bounds SOC is genuinely unstable."
            )
        if not getattr(ct, "h2_fraction_ok", True):
            lines.append(
                "  * H2 fraction violated: keep h2_cost pure proportional "
                "and ensure it remains a meaningful share of objective_cost."
            )
    lines.append("Please revise get_reward(self) to address every violation listed above.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_agent_review(
    *,
    args,
    client,
    feedback_bundle: Dict[str, str],
    current_method: str,
    iter_dir: Path,
    chunk_result,
    observer,
    reward_history=None,
    prev_method: Optional[str] = None,
    iteration_memory: Optional[Dict[str, Any]] = None,
    policy_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Full multi-agent review pipeline (six-priority multi-agentic workflow).

    Returns the same dict shape as _run_llm_with_tpe_retry().
    Extra keys: mar_decision, diagnostic_result, mediation_result, counterfactual_result.
    iteration_memory: {trajectory: {...}, insights: [...]} for structured memory (Priority 5).
    policy_quality: Python policy score dict from main loop (Priority 6).
    """
    review_dir = iter_dir / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    usage_records: List[Dict[str, Any]] = []
    usage_path = iter_dir / "llm_usage.json"

    # ── Stage 1: context builder ──────────────────────────────────────────────
    observer.status("MAR_CONTEXT", "building review_packet")
    historical_guard = _build_historical_reward_guard_payload(reward_history, current_method)
    review_packet = build_review_packet(
        feedback_bundle=feedback_bundle,
        current_method=current_method,
        chunk_result=chunk_result,
        args=args,
        iteration=observer.iteration,
        iteration_memory=iteration_memory,
        policy_quality=policy_quality,
    )
    review_packet["historical_reward_guard"] = historical_guard
    _write_json(iter_dir / "review_packet.json", review_packet)
    _write_json(iter_dir / "historical_reward_guard.json", historical_guard)

    # ── Tool context (auto-constructed for Anthropic; None for other providers) ─
    tool_ctx: Optional[ToolContext] = None
    if getattr(client, "provider", "") == "anthropic":
        tool_ctx = ToolContext(
            log_path=str(chunk_result.log_path),
            episode_data_dir=(
                str(chunk_result.episode_data_dir) if chunk_result.episode_data_dir else None
            ),
            reward_file=str(getattr(args, "reward_file", "")),
            class_name=str(getattr(args, "class_name", "EMS")),
            method_name=str(getattr(args, "method_name", "get_reward")),
            args=args,
            required_info_keys=list(REQUIRED_INFO_KEYS),
            run_dir=str(iter_dir.parent),
            current_iteration=observer.iteration,
        )

    # ── Stage 1.5: diagnostic agent (Python — Priority 1) ────────────────────
    observer.status("MAR_DIAGNOSTIC", "classifying training failure mode (Python)")
    diagnostic_result = _compute_diagnostic_from_packet(review_packet)
    review_packet["diagnostic_result"] = diagnostic_result
    _write_json(review_dir / "diagnostic_agent.json", {
        k: v for k, v in diagnostic_result.items() if not k.startswith("_")
    })
    observer.status(
        "MAR_DIAGNOSTIC_DONE",
        f"failure_mode={diagnostic_result['failure_mode']} "
        f"convergence={diagnostic_result['convergence_quality']} "
        f"oob={diagnostic_result['oob_rate']:.1%} "
        f"swing={diagnostic_result['mean_soc_swing']}",
    )

    # ── Stage 2a: five LLM reviewers — round 0 (independent opinions) ───────
    reviewer_outputs: Dict[str, Dict[str, Any]] = {}
    llm_reviewer_order = [
        "soc_safety", "fuel_efficiency", "reward_hacking",
        "fc_durability", "battery_usage",  # Priority 2 specialists
    ]
    blackboard_posts: List[Dict[str, Any]] = []
    conflict_signals: List[Dict[str, Any]] = []
    debate_thread: List[Dict[str, Any]] = []
    for reviewer_name in llm_reviewer_order:
        agent_model = get_agent_model(client.provider, reviewer_name, client.model)
        observer.status(
            f"MAR_{reviewer_name.upper()}",
            f"running {reviewer_name} [round 0, {agent_model}]",
        )
        result = _run_reviewer(
            client,
            reviewer_name,
            review_packet,
            args,
            usage_records,
            usage_path,
            model=agent_model,
            ctx=tool_ctx,
        )
        reviewer_outputs[reviewer_name] = result
        file_payload = {k: v for k, v in result.items() if not k.startswith("_")}
        _write_json(review_dir / f"{reviewer_name}.json", file_payload)
        _write_text(review_dir / f"{reviewer_name}_response.md", result.get("_raw_response", ""))
        observer.status(
            f"MAR_{reviewer_name.upper()}_DONE",
            f"decision={result.get('decision', '?')} risk={result.get('risk_score', '?')}",
        )
        # ① Collect blackboard posts for insight_memory
        bp = result.get("blackboard_post")
        if bp and isinstance(bp, str) and bp.strip():
            blackboard_posts.append({
                "type": "blackboard_post",
                "agent": reviewer_name,
                "observation": bp.strip(),
            })
        # ② Collect conflict signals for mediator
        cs = result.get("conflict_signal")
        if cs and isinstance(cs, dict) and cs.get("against") and cs.get("issue"):
            conflict_signals.append({
                "from": reviewer_name,
                "against": cs["against"],
                "issue": cs["issue"],
            })
        # Seed the debate thread for DEBATE_AGENTS
        if reviewer_name in DEBATE_AGENTS:
            debate_thread.append({
                "sender": reviewer_name,
                "content": json.dumps(file_payload, ensure_ascii=False),
                "round_idx": 0,
            })

    # ── Stage 2a.1: peer-rebuttal rounds (Condition 1) ────────────────────────
    # debate_rounds=1 means round 0 only (no rebuttal). >=2 adds rebuttal rounds.
    debate_rounds = max(1, int(getattr(args, "debate_rounds", 1)))
    for round_idx in range(1, debate_rounds):
        observer.status(
            f"MAR_DEBATE_ROUND_{round_idx}",
            f"peer-rebuttal round {round_idx}: {DEBATE_AGENTS}",
        )
        round_snapshot = list(debate_thread)  # opinions visible at start of this round
        for reviewer_name in DEBATE_AGENTS:
            agent_model = get_agent_model(client.provider, reviewer_name, client.model)
            observer.status(
                f"MAR_{reviewer_name.upper()}_R{round_idx}",
                f"rebuttal round {round_idx}: {reviewer_name} [{agent_model}]",
            )
            peer_thread = [m for m in round_snapshot if m["sender"] != reviewer_name]
            result = _run_reviewer(
                client,
                reviewer_name,
                review_packet,
                args,
                usage_records,
                usage_path,
                model=agent_model,
                debate_round=round_idx,
                debate_thread=peer_thread,
                ctx=tool_ctx,
            )
            reviewer_outputs[reviewer_name] = result  # overwrite with rebuttal opinion
            file_payload = {k: v for k, v in result.items() if not k.startswith("_")}
            _write_json(review_dir / f"{reviewer_name}_round{round_idx}.json", file_payload)
            _write_text(
                review_dir / f"{reviewer_name}_round{round_idx}_response.md",
                result.get("_raw_response", ""),
            )
            observer.status(
                f"MAR_{reviewer_name.upper()}_R{round_idx}_DONE",
                f"decision={result.get('decision', '?')} risk={result.get('risk_score', '?')}",
            )
            # Update debate_thread with this round's opinion
            debate_thread = [m for m in debate_thread if m["sender"] != reviewer_name]
            debate_thread.append({
                "sender": reviewer_name,
                "content": json.dumps(file_payload, ensure_ascii=False),
                "round_idx": round_idx,
            })

    # ── Stage 2b: scale reviewer (pure Python, no LLM) ───────────────────────
    observer.status("MAR_SCALE", "computing scale check (Python)")
    scale_result = _compute_scale_check_from_packet(review_packet, prev_source=prev_method)
    reviewer_outputs["scale_reviewer"] = scale_result
    _write_json(review_dir / "scale_reviewer.json", {
        k: v for k, v in scale_result.items() if not k.startswith("_")
    })
    observer.status(
        "MAR_SCALE_DONE",
        f"decision={scale_result['decision']} "
        f"violations={len(scale_result['violations'])} "
        f"prescribed={scale_result['prescribed_corrections']}",
    )

    # ── Stage 2c: trainability reviewer (pure Python, no LLM) ────────────────
    observer.status("MAR_TRAINABILITY", "computing trainability (Python)")
    trainability_result = _compute_trainability_from_packet(review_packet)
    reviewer_outputs["trainability_scheduler"] = trainability_result
    _write_json(review_dir / "trainability_scheduler.json", {
        k: v for k, v in trainability_result.items() if not k.startswith("_")
    })
    observer.status(
        "MAR_TRAINABILITY_DONE",
        f"decision={trainability_result['decision']} "
        f"lr_factor={trainability_result['lr_proposal']['lr_factor']}",
    )

    # ── Stage 2.5: trade-off mediator (LLM — Priority 3) ────────────────────���
    mediator_model = get_agent_model(client.provider, "tradeoff_mediator", client.model)
    observer.status("MAR_MEDIATOR", f"running trade-off mediator [{mediator_model}]")
    mediation_result = _run_tradeoff_mediator(
        client,
        reviewer_outputs,
        diagnostic_result,
        args,
        usage_records,
        usage_path,
        model=mediator_model,
        debate_thread=debate_thread,
        conflict_signals=conflict_signals,
        ctx=tool_ctx,
    )
    _write_json(review_dir / "tradeoff_mediator.json", {
        k: v for k, v in mediation_result.items() if not k.startswith("_")
    })
    _write_text(review_dir / "tradeoff_mediator_response.md", mediation_result.get("_raw_response", ""))
    observer.status(
        "MAR_MEDIATOR_DONE",
        f"conflict={mediation_result.get('conflict_detected', False)} "
        f"h2_suppression={mediation_result.get('h2_suppression_risk_active', False)} "
        f"priority={mediation_result.get('mediated_priority', '?')} "
        f"blocked={len(mediation_result.get('blocked_changes', []))}",
    )

    # Inject mediator result into reviewer_outputs so orchestrator sees it
    reviewer_outputs["tradeoff_mediator"] = {
        k: v for k, v in mediation_result.items() if not k.startswith("_")
    }

    # ── Stage 3: orchestrator ─────────────────────────────────────────────────
    orchestrator_model = get_agent_model(client.provider, "orchestrator", client.model)
    observer.status("MAR_ORCHESTRATOR", f"running orchestrator [{orchestrator_model}]")
    orchestrator_spec = _run_orchestrator(
        client,
        reviewer_outputs,
        historical_guard,
        args,
        usage_records,
        usage_path,
        model=orchestrator_model,
        ctx=tool_ctx,
    )
    _write_json(
        iter_dir / "orchestrator_spec.json",
        {k: v for k, v in orchestrator_spec.items() if not k.startswith("_")},
    )
    _write_text(iter_dir / "orchestrator_response.md", orchestrator_spec.get("_raw_response", ""))

    overall_decision = orchestrator_spec.get("overall_decision", "revise")
    # Safety: 'reject' skips the proposer entirely and keeps the broken reward unchanged.
    # Always fall back to 'revise' so the proposer gets a chance to repair the violations.
    if overall_decision == "reject":
        observer.status(
            "MAR_REJECT_OVERRIDE",
            "orchestrator returned 'reject'; overriding to 'revise' so proposer can repair",
        )
        orchestrator_spec["overall_decision"] = "revise"
        overall_decision = "revise"
    observer.status("MAR_ORCHESTRATOR_DONE", f"overall_decision={overall_decision}")

    # ── Stage 3.5: counterfactual risk check (Python — Priority 4) ───────────
    observer.status("MAR_COUNTERFACTUAL", "checking orchestrator_spec for high-risk changes (Python)")
    cf_result = _check_counterfactual_risks(
        orchestrator_spec, diagnostic_result, reviewer_outputs, mediation_result
    )
    _write_json(review_dir / "counterfactual_evaluator.json", {
        k: v for k, v in cf_result.items() if not k.startswith("_")
    })
    observer.status(
        "MAR_COUNTERFACTUAL_DONE",
        f"overall_risk={cf_result['overall_risk']} "
        f"flags={len(cf_result['risk_flags'])} "
        f"safe={cf_result['must_change_safe']} "
        f"appeal={cf_result.get('appeal_flag', False)}",
    )

    # ── Stage 3.6: orchestrator appeal (Condition 2 — bidirectional) ─────────
    # Triggered when counterfactual finds high-risk changes the mediator did not block.
    # Re-runs orchestrator with appeal context; re-runs counterfactual once (no further appeal).
    if cf_result.get("appeal_flag"):
        observer.status(
            "MAR_APPEAL",
            f"counterfactual appeal → re-running orchestrator [{orchestrator_model}] "
            f"(grounds: {cf_result['appeal_grounds'][:120]}...)",
        )
        appeal_ctx = _ORCHESTRATOR_APPEAL_SUFFIX.replace(
            "__APPEAL_GROUNDS__", cf_result["appeal_grounds"]
        )
        orchestrator_spec = _run_orchestrator(
            client,
            reviewer_outputs,
            historical_guard,
            args,
            usage_records,
            usage_path,
            model=orchestrator_model,
            appeal_context=appeal_ctx,
            ctx=tool_ctx,
        )
        # Preserve reject-override logic for the appeal spec as well
        if orchestrator_spec.get("overall_decision") == "reject":
            orchestrator_spec["overall_decision"] = "revise"
        _write_json(
            iter_dir / "orchestrator_spec_appeal.json",
            {k: v for k, v in orchestrator_spec.items() if not k.startswith("_")},
        )
        _write_text(
            iter_dir / "orchestrator_response_appeal.md",
            orchestrator_spec.get("_raw_response", ""),
        )
        # Re-run counterfactual with the amended spec (appeal_flag suppressed on second run)
        cf_result = _check_counterfactual_risks(
            orchestrator_spec, diagnostic_result, reviewer_outputs, mediation_result
        )
        cf_result["_appeal_round"] = 2
        _write_json(review_dir / "counterfactual_evaluator_appeal.json", {
            k: v for k, v in cf_result.items() if not k.startswith("_")
        })
        observer.status(
            "MAR_APPEAL_DONE",
            f"overall_risk={cf_result['overall_risk']} "
            f"flags={len(cf_result['risk_flags'])} "
            f"safe={cf_result['must_change_safe']}",
        )

    # ── Stage 4: proposer with aligned-gate retry loop ───────────────────────
    max_tpe_attempts = 1 + max(0, int(getattr(args, "tpe_retries", 3)))
    max_parse_attempts = 1 + max(0, int(getattr(args, "parse_retries", max_tpe_attempts)))

    proposer_model = get_agent_model(client.provider, "proposer", client.model)
    proposer_prompt = _build_proposer_prompt(orchestrator_spec, current_method, historical_guard)
    # Append counterfactual warnings to proposer prompt when high-risk changes are flagged
    if cf_result["warning_lines"]:
        proposer_prompt = (
            proposer_prompt
            + "\n\n# Counterfactual Risk Warnings (must respect — Priority 4)\n"
            + "\n".join(cf_result["warning_lines"])
            + "\n"
        )
    _write_text(iter_dir / "proposer_prompt.md", proposer_prompt)

    parse_ok_count = 0
    parse_fail_count = 0
    historical_duplicate_count = 0
    tpe_eval_count = 0
    best_tpe_delta: Optional[float] = None
    best_candidate_method: Optional[str] = None
    best_candidate_attempt: Optional[int] = None
    passed_candidate_method: Optional[str] = None
    passed_tpe = None
    current_proposer_prompt = proposer_prompt

    attempt = 1
    while attempt <= max_parse_attempts and tpe_eval_count < max_tpe_attempts:
        observer.status(
            f"MAR_PROPOSER_{attempt}",
            f"requesting proposer candidate [{proposer_model}]",
        )

        proposer_tools = get_tools_for_agent("proposer") if tool_ctx is not None else []
        proposer_exec = (
            (lambda name, inp: execute_tool(name, inp, tool_ctx, agent_name="proposer"))
            if tool_ctx is not None else None
        )
        response = _tracked_complete_with_tools(
            client=client,
            usage_records=usage_records,
            usage_path=usage_path,
            stage=f"proposer:attempt_{attempt}",
            prompt=current_proposer_prompt,
            tools=proposer_tools,
            tool_executor=proposer_exec,
            temperature=float(getattr(args, "llm_temperature", 0.2)),
            max_tokens=int(getattr(args, "llm_max_tokens", 3500)),
            timeout_sec=int(getattr(args, "llm_timeout_sec", 240)),
            model=proposer_model,
            ctx=tool_ctx,
        )
        _write_text(iter_dir / f"proposer_response_attempt_{attempt}.md", response)

        # ── parse ──
        try:
            candidate_method = extract_method_from_llm_response(response, method_name="get_reward")
        except Exception as e:
            parse_fail_count += 1
            observer.status(f"MAR_PARSE_FAIL_{attempt}", str(e))
            if attempt < max_parse_attempts and tpe_eval_count < max_tpe_attempts:
                current_proposer_prompt = (
                    current_proposer_prompt
                    + "\n\n# Parse Error (fix immediately)\n"
                    + f"Previous response parse failed: {e}\n"
                    + "Output MUST include a valid Python ```python ... ``` block "
                    + "containing only def get_reward(self):."
                )
                _write_text(iter_dir / f"proposer_prompt_attempt_{attempt + 1}.md", current_proposer_prompt)
            attempt += 1
            continue

        candidate_method = ensure_logging_contract(candidate_method, REQUIRED_INFO_KEYS)
        parse_ok_count += 1
        _write_text(iter_dir / f"candidate_get_reward_attempt_{attempt}.py", candidate_method)

        candidate_signature = _method_signature(candidate_method)
        history_entry = (reward_history or {}).get(candidate_signature)
        if history_entry is not None:
            historical_duplicate_count += 1
            matched_label = _history_match_label(history_entry)
            observer.status(
                "MAR_HISTORICAL_DUPLICATE",
                f"attempt {attempt}: candidate matches trained reward {matched_label}",
            )
            _write_json(
                iter_dir / f"historical_duplicate_attempt_{attempt}.json",
                {
                    "candidate_signature": candidate_signature,
                    "matched_history": matched_label,
                },
            )
            if attempt < max_parse_attempts and tpe_eval_count < max_tpe_attempts:
                current_proposer_prompt = (
                    current_proposer_prompt
                    + "\n\n# Historical Reward Rejection (must address)\n"
                    + f"Previous candidate structurally matched an already-trained reward: {matched_label}.\n"
                    + "Do not return the current reward unchanged.\n"
                    + "Do not revert to any previously trained reward variant.\n"
                    + "Propose a genuinely new local refinement to get_reward(self).\n"
                )
                _write_text(
                    iter_dir / f"proposer_prompt_attempt_{attempt + 1}.md",
                    current_proposer_prompt,
                )
            attempt += 1
            continue

        key_changes = infer_key_weight_changes(current_method, candidate_method)
        _write_json(
            iter_dir / f"key_weight_changes_attempt_{attempt}.json",
            {"changes": key_changes},
        )

        # ── aligned gate ──
        from ldri_aligned_gate import evaluate_aligned_gate
        tpe_eval_count += 1
        aligned = evaluate_aligned_gate(
            candidate_source=candidate_method,
            log_path=chunk_result.log_path,
            episode_data_dir=chunk_result.episode_data_dir,
            args=args,
            min_tac=float(getattr(args, "aligned_min_tac", 0.3)),
            dominance_threshold=float(getattr(args, "aligned_dominance_threshold", 0.95)),
            soc_dominance_threshold=float(getattr(args, "aligned_soc_dominance_threshold", 0.85)),
            min_h2_fraction=float(getattr(args, "aligned_min_h2_fraction", 0.25)),
            h2_perturbation_ratio=float(getattr(args, "aligned_h2_perturbation_ratio", 0.2)),
        )
        _write_json(iter_dir / f"aligned_gate_attempt_{attempt}.json", {
            "passed": aligned.passed,
            "tac_passed": aligned.tac.passed,
            "tac_skipped": aligned.tac.skipped,
            "tac_tau": aligned.tac.tau if (aligned.tac.tau is not None and aligned.tac.tau == aligned.tac.tau) else None,
            "tac_n": aligned.tac.n_episodes,
            "tac_threshold": aligned.tac.threshold,
            "tac_h2_available": aligned.tac.h2_available,
            "constitution_passed": aligned.constitution.passed,
            "constitution_skipped": aligned.constitution.skipped,
            "constitution_h2_mono": aligned.constitution.h2_monotonicity_ok,
            "constitution_soc_bnd": aligned.constitution.soc_boundary_ok,
            "constitution_dominance": aligned.constitution.dominance_ok,
            "constitution_soc_dominance": aligned.constitution.soc_dominance_ok,
            "constitution_h2_fraction": aligned.constitution.h2_fraction_ok,
            "message": aligned.message,
        })
        delta: Optional[float] = (
            float(aligned.tac.tau) if aligned.tac.tau is not None else None
        )
        if delta is not None and (best_tpe_delta is None or delta > best_tpe_delta):
            best_tpe_delta = delta
            best_candidate_method = candidate_method
            best_candidate_attempt = attempt
        elif best_candidate_method is None:
            best_candidate_method = candidate_method
            best_candidate_attempt = attempt
        status_tag = "MAR_GATE_PASS" if aligned.passed else "MAR_GATE_FAIL"
        observer.status(status_tag, f"attempt {attempt}: {aligned.message}")
        if aligned.passed:
            passed_candidate_method = candidate_method
            passed_tpe = aligned
            break
        if attempt < max_parse_attempts and tpe_eval_count < max_tpe_attempts:
            current_proposer_prompt = (
                current_proposer_prompt + _build_aligned_gate_feedback(aligned)
            )
            _write_text(iter_dir / f"proposer_prompt_attempt_{attempt + 1}.md", current_proposer_prompt)

        attempt += 1

    # ── Stage 5: verification summary ────────────────────────────────────────
    _write_json(iter_dir / "proposal_verification.json", {
        "overall_decision": overall_decision,
        "parse_ok_count": parse_ok_count,
        "parse_fail_count": parse_fail_count,
        "historical_duplicate_count": historical_duplicate_count,
        "tpe_eval_count": tpe_eval_count,
        "best_tpe_delta": best_tpe_delta,
        "best_candidate_attempt": best_candidate_attempt,
        "passed": passed_candidate_method is not None,
        "diagnostic_failure_mode": diagnostic_result.get("failure_mode"),
        "mediation_conflict": mediation_result.get("conflict_detected", False),
        "counterfactual_risk": cf_result.get("overall_risk", "low"),
    })
    _write_llm_usage_log(usage_path, usage_records)

    return {
        "parse_ok_count": parse_ok_count,
        "parse_fail_count": parse_fail_count,
        "historical_duplicate_count": historical_duplicate_count,
        "tpe_eval_count": tpe_eval_count,
        "max_parse_attempts": max_parse_attempts,
        "max_tpe_attempts": max_tpe_attempts,
        "best_tpe_delta": best_tpe_delta,
        "best_candidate_method": best_candidate_method,
        "best_candidate_attempt": best_candidate_attempt,
        "passed_candidate_method": passed_candidate_method,
        "passed_tpe": passed_tpe,
        "mar_decision": overall_decision,
        "trainability_lr_proposal": trainability_result.get("lr_proposal", {}),
        "llm_usage_totals": _llm_usage_totals(usage_records),
        "diagnostic_result": {k: v for k, v in diagnostic_result.items() if not k.startswith("_")},
        "mediation_result": {k: v for k, v in mediation_result.items() if not k.startswith("_")},
        "counterfactual_result": {k: v for k, v in cf_result.items() if not k.startswith("_")},
        "reviewer_blackboard_posts": blackboard_posts,
    }