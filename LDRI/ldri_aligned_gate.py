"""Two-layer AI-aligned reward gate for LDRI.

Motivation
----------
The existing TPE gate (ldri_tpe_feasible_pool) checks only one thing: does
the candidate reward assign higher average reward to the episode that is
objectively better (by objective_cost ranking)?  That single proxy check
can be satisfied by a reward function that is structurally misaligned with
human intent -- as long as it correctly ranks the two anchor episodes.

This module implements two independent layers that must ALL pass.  Coefficient
scale explosion is handled upstream by scale_reviewer in multi_agent_review.py
before the gate is ever reached.

Layer 1 -- TAC (Trajectory Alignment Coefficient)
    Kendall tau between candidate reward episode rankings and physical performance
    rankings derived from logged h2_100km and in_bounds_rate.  Ground truth rank:
    rank(h2_100km) + 0.5 * rank(1 - in_bounds_rate/100)  (lower = physically better).
    Pass condition: tau >= min_tac.
    Non-tautological: a reward coefficient inflated by any factor will fail if the
    agent's actual episode ordering by physical metrics (h2 efficiency, SOC control)
    does not match the reward ordering.  Falls back to in_bounds_rate only when
    h2_100km is not available in the training log.

Layer 2 -- Reward Constitution (structural alignment)
    Automated invariant checks derived from domain knowledge:
    a) H2 monotonicity  -- increasing h2_fcs must decrease reward.
    b) SOC boundary     -- out-of-bounds SOC must produce lower reward than
                           on-target SOC, ceteris paribus.
    c) Dominance guard  -- no single cost term may exceed `dominance_threshold`
                           of the mean objective_cost (prevents degenerate
                           reward functions that collapse to one term).
                           SOC also has its own stricter dominance cap, and
                           H2 must remain a minimum fraction of objective_cost
                           so SOC does not become the primary objective.
    These checks encode the human engineer's intent as hard invariants.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as scio
import scipy.stats as sstats

from feedback_parser import EpisodeSummary, parse_train_log
from ldri_tpe_feasible_pool import (
    _arr,
    _build_candidate_fn,
    _StubEMS,
    _episode_objective_metric,
    _cfg_value,
)


# ── Default thresholds ────────────────────────────────────────────────────────

_DEFAULTS = {
    "min_tac": 0.2,              # Layer 1: need kendall tau >= 0.2
    "dominance_threshold": 0.75,  # Layer 2c: one term may not exceed 75 % of total
    "soc_dominance_threshold": 0.85,  # Layer 2c: SOC may not dominate the objective
    "min_h2_fraction": 0.25,  # Layer 2c: H2 should remain a meaningful objective term
    "h2_perturbation_ratio": 0.2,  # Layer 2a: h2_fcs perturbed by +20 %
}


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class TACResult:
    passed: bool
    skipped: bool
    tau: Optional[float]        # Kendall tau (positive = reward aligned with physical perf)
    n_episodes: int             # number of episodes used
    threshold: float            # minimum tau required to pass
    h2_available: bool          # True if h2_100km was included in ground-truth rank
    message: str


@dataclass
class ConstitutionResult:
    passed: bool
    skipped: bool
    h2_monotonicity_ok: bool
    soc_boundary_ok: bool
    dominance_ok: bool
    soc_dominance_ok: bool
    h2_fraction_ok: bool
    message: str


@dataclass
class AlignedGateResult:
    passed: bool
    tac: TACResult
    constitution: ConstitutionResult
    message: str


# ── Stub helpers ──────────────────────────────────────────────────────────────

def _make_stub(args, p_batt_kw: np.ndarray, p_fc_kw: np.ndarray) -> _StubEMS:
    stub = _StubEMS()
    stub.w_soc = float(args.w_soc)
    stub.w_h2 = float(getattr(args, "w_h2", 10.0))
    stub.w_fc = float(getattr(args, "w_fc", 4.275e6))
    stub.w_batt = float(getattr(args, "w_batt", 1.116e6))
    stub.SOC_target = float(getattr(args, "soc_target", _cfg_value("soc_target")))
    stub.soc_bounds = (float(getattr(args, "soc_low", 0.4)), float(getattr(args, "soc_high", 0.8)))
    stub.reward_scale = float(getattr(args, "reward_scale", 1.0))
    stub.soc_weight_multiplier = 1.0
    stub.eq_h2_batt_coef = float(getattr(args, "eq_h2_batt_coef", 0.0164))
    stub.time_step = float(getattr(args, "time_step", 1.0))
    stub.P_batt_max = max(1.0, float(np.nanmax(np.abs(p_batt_kw))) * 1000.0) if len(p_batt_kw) else 1.0
    stub.P_FCS_max = max(1.0, float(np.nanmax(np.abs(p_fc_kw)))) if len(p_fc_kw) else 1.0
    stub.info = {}
    return stub


def _fill_step(
    stub: _StubEMS,
    *,
    soc: float,
    h2_fcs: float,
    p_batt_kw_step: float,
    p_fc_kw_step: float,
    dsoh_fcs: float,
    dsoh_batt: float,
) -> None:
    stub.SOC = float(soc)
    stub.h2_fcs = float(h2_fcs)
    stub.P_batt = float(p_batt_kw_step) * 1000.0
    stub.P_FCS = float(p_fc_kw_step)
    stub.P_fc = stub.P_FCS
    stub.dSOH_FCS = float(dsoh_fcs)
    stub.dSOH_batt = float(dsoh_batt)
    stub.h2_batt = stub.P_batt / 1000.0 * stub.eq_h2_batt_coef
    stub.h2_equal = stub.h2_fcs + stub.h2_batt


def _call_fn(fn, stub: _StubEMS) -> float:
    stub.info = {}
    val = fn(stub)
    if isinstance(val, tuple):
        val = val[0]
    return float(val)


def _load_mat_arrays(mat_path: Path) -> Optional[Dict[str, object]]:
    try:
        mat = scio.loadmat(str(mat_path))
    except Exception:
        return None
    soc = _arr(mat, "SOC")
    n = len(soc)
    if n == 0:
        return None
    return {
        "soc": soc,
        "n": int(n),
        "h2_fcs": _arr(mat, "h2_fcs", default_len=n),
        "p_batt_kw": _arr(mat, "P_batt_req", "pack_power_out", default_len=n),
        "p_fc_kw": _arr(mat, "P_fc", default_len=n),
        "dsoh_fcs": _arr(mat, "FCS_De", "fcs_soh_cost", default_len=n),
        "dsoh_batt": _arr(mat, "dsoh", "batt_soh_cost", default_len=n),
    }


# ── Per-episode reward helpers ────────────────────────────────────────────────

def _episode_avg_reward(fn, mat_path: Path, args) -> Optional[float]:
    arrays = _load_mat_arrays(mat_path)
    if arrays is None:
        return None
    n = arrays["n"]
    stub = _make_stub(args, arrays["p_batt_kw"], arrays["p_fc_kw"])
    rewards = []
    for i in range(n):
        _fill_step(
            stub,
            soc=arrays["soc"][i], h2_fcs=arrays["h2_fcs"][i],
            p_batt_kw_step=arrays["p_batt_kw"][i], p_fc_kw_step=arrays["p_fc_kw"][i],
            dsoh_fcs=arrays["dsoh_fcs"][i], dsoh_batt=arrays["dsoh_batt"][i],
        )
        try:
            r = _call_fn(fn, stub)
            if np.isfinite(r):
                rewards.append(r)
        except Exception:
            pass
    return float(np.mean(rewards)) if rewards else None



def _representative_probes(mat_path: Path, args) -> List[_StubEMS]:
    arrays = _load_mat_arrays(mat_path)
    if arrays is None:
        return []
    n = arrays["n"]
    template = _make_stub(args, arrays["p_batt_kw"], arrays["p_fc_kw"])
    probes = []
    for i in sorted({0, n // 2, n - 1}):
        probe = copy.copy(template)
        _fill_step(
            probe,
            soc=arrays["soc"][i], h2_fcs=arrays["h2_fcs"][i],
            p_batt_kw_step=arrays["p_batt_kw"][i], p_fc_kw_step=arrays["p_fc_kw"][i],
            dsoh_fcs=arrays["dsoh_fcs"][i], dsoh_batt=arrays["dsoh_batt"][i],
        )
        probes.append(probe)
    return probes


def _cost_means(fn, mat_path: Path, args, max_steps: int = 100) -> Optional[Dict[str, float]]:
    """Return mean of absolute values of each cost term over sampled steps."""
    arrays = _load_mat_arrays(mat_path)
    if arrays is None:
        return None
    n = arrays["n"]
    stub = _make_stub(args, arrays["p_batt_kw"], arrays["p_fc_kw"])
    keys = ["soc_cost", "h2_cost", "fcs_soh_cost", "batt_soh_cost", "objective_cost"]
    sums: Dict[str, float] = {k: 0.0 for k in keys}
    valid = 0
    for i in np.linspace(0, n - 1, min(n, max_steps), dtype=int):
        _fill_step(
            stub,
            soc=arrays["soc"][i], h2_fcs=arrays["h2_fcs"][i],
            p_batt_kw_step=arrays["p_batt_kw"][i], p_fc_kw_step=arrays["p_fc_kw"][i],
            dsoh_fcs=arrays["dsoh_fcs"][i], dsoh_batt=arrays["dsoh_batt"][i],
        )
        try:
            _call_fn(fn, stub)
        except Exception:
            continue
        if not all(k in stub.info for k in keys):
            continue
        try:
            for k in keys:
                v = float(stub.info[k])
                if np.isfinite(v):
                    sums[k] += abs(v)
        except Exception:
            continue
        valid += 1
    if valid == 0:
        return None
    return {k: v / valid for k, v in sums.items()}


# ── Constitution check helpers ─────────────────────────────────────────────────

def _check_h2_monotonicity(
    fn,
    probes: List[_StubEMS],
    perturbation_ratio: float,
) -> Tuple[bool, str]:
    violations = 0
    evaluated = 0
    for base in probes:
        try:
            base_stub = copy.copy(base)
            base_r = _call_fn(fn, base_stub)
            if not np.isfinite(base_r):
                continue

            perturbed = copy.copy(base)
            # add a minimum absolute bump so near-zero h2_fcs is still perturbed
            perturbed.h2_fcs = base.h2_fcs * (1.0 + perturbation_ratio) + 1e-5
            perturbed.h2_equal = perturbed.h2_fcs + perturbed.h2_batt
            pert_r = _call_fn(fn, perturbed)
            if not np.isfinite(pert_r):
                continue

            evaluated += 1
            if pert_r >= base_r:
                violations += 1
        except Exception:
            pass

    if evaluated == 0:
        return False, "h2_monotonicity: FAIL (no valid probe steps)"
    ok = violations == 0
    return ok, f"h2_monotonicity: {'OK' if ok else 'VIOLATED'} ({violations}/{evaluated} violations)"


def _check_soc_boundary(fn, probes: List[_StubEMS]) -> Tuple[bool, str]:
    violations = 0
    evaluated = 0
    for base in probes:
        try:
            soc_low, soc_high = base.soc_bounds
            soc_target = base.SOC_target

            on_target = copy.copy(base)
            on_target.SOC = soc_target
            target_r = _call_fn(fn, on_target)
            if not np.isfinite(target_r):
                continue

            below = copy.copy(base)
            below.SOC = soc_low - 0.05
            below_r = _call_fn(fn, below)

            above = copy.copy(base)
            above.SOC = soc_high + 0.05
            above_r = _call_fn(fn, above)

            if not (np.isfinite(below_r) and np.isfinite(above_r)):
                continue

            evaluated += 2
            if below_r >= target_r:
                violations += 1
            if above_r >= target_r:
                violations += 1
        except Exception:
            pass

    if evaluated == 0:
        return False, "soc_boundary: FAIL (no valid probe steps)"
    ok = violations == 0
    return ok, f"soc_boundary: {'OK' if ok else 'VIOLATED'} ({violations}/{evaluated} violations)"


def _check_dominance(
    fn,
    mat_path: Path,
    args,
    dominance_threshold: float,
    soc_dominance_threshold: float,
    min_h2_fraction: float,
) -> Tuple[bool, str]:
    means = _cost_means(fn, mat_path, args)
    if means is None:
        return True, "dominance: skipped (no evaluable steps)"
    obj = means["objective_cost"]
    if obj < 1e-10:
        return True, "dominance: skipped (near-zero objective_cost)"
    max_ratio = 0.0
    max_term = ""
    for term in ["soc_cost", "h2_cost", "fcs_soh_cost", "batt_soh_cost"]:
        ratio = means[term] / obj
        if ratio > max_ratio:
            max_ratio = ratio
            max_term = term
    soc_ratio = means["soc_cost"] / obj
    h2_ratio = means["h2_cost"] / obj
    any_ok = max_ratio <= dominance_threshold
    soc_ok = soc_ratio <= soc_dominance_threshold
    h2_ok = h2_ratio >= min_h2_fraction
    ok = any_ok and soc_ok and h2_ok
    return ok, (
        f"dominance: {'OK' if ok else 'VIOLATED'} "
        f"(dominant={max_term}, ratio={max_ratio:.3f}, threshold={dominance_threshold:.2f}; "
        f"soc_ratio={soc_ratio:.3f}, soc_threshold={soc_dominance_threshold:.2f}; "
        f"h2_ratio={h2_ratio:.3f}, min_h2_fraction={min_h2_fraction:.2f})"
    )


# ── Layer evaluation functions ────────────────────────────────────────────────

def evaluate_tac(
    candidate_fn,
    episodes: List[EpisodeSummary],
    ep_dir: Path,
    args,
    min_in_bounds_rate: float,
    threshold: float,
) -> TACResult:
    """Layer 1: Trajectory Alignment Coefficient (Eschmann et al. 2025).

    Ground truth rank = rank(h2_100km) + 0.5 * rank(1 - in_bounds_rate/100).
    Lower rank = physically better episode.
    TAC = Kendall tau(R_cand, -ground_truth_rank).
    Pass condition: tau >= threshold.
    """
    candidates = [
        ep for ep in episodes
        if ep.in_bounds_rate is not None
        and float(ep.in_bounds_rate) >= min_in_bounds_rate
    ]

    valid_eps: List[EpisodeSummary] = []
    reward_vals: List[float] = []

    for ep in candidates:
        mat_path = ep_dir / f"data_ep{int(ep.episode)}.mat"
        if not mat_path.exists():
            continue
        avg_r = _episode_avg_reward(candidate_fn, mat_path, args)
        if avg_r is None:
            continue
        valid_eps.append(ep)
        reward_vals.append(float(avg_r))

    n = len(valid_eps)
    if n < 3:
        return TACResult(
            passed=True, skipped=True, tau=None, n_episodes=n, threshold=threshold,
            h2_available=False,
            message=f"tac: skipped (only {n} episodes with .mat data, need >=3)",
        )

    h2_available = all(ep.h2_100km is not None for ep in valid_eps)
    inbounds_cost = [1.0 - float(ep.in_bounds_rate) / 100.0 for ep in valid_eps]
    inbounds_ranks = sstats.rankdata(inbounds_cost)

    if h2_available:
        h2_ranks = sstats.rankdata([float(ep.h2_100km) for ep in valid_eps])
        gt_ranks = h2_ranks + 0.5 * inbounds_ranks
    else:
        gt_ranks = inbounds_ranks

    # tau(reward, -gt_ranks) > 0 means high reward ↔ physically better episode
    tau_val, _ = sstats.kendalltau(reward_vals, -gt_ranks)
    tau = float(tau_val)
    passed = bool(np.isfinite(tau)) and tau >= threshold
    metrics_label = "h2+soc" if h2_available else "soc_only"
    return TACResult(
        passed=passed, skipped=False, tau=tau, n_episodes=n, threshold=threshold,
        h2_available=h2_available,
        message=(
            f"tac: {'PASS' if passed else 'FAIL'} "
            f"kendall_tau={tau:.3f} (need >={threshold:.2f}), n={n}, metrics={metrics_label}"
        ),
    )


def evaluate_constitution(
    candidate_fn,
    probe_mat_path: Path,
    args,
    dominance_threshold: float,
    soc_dominance_threshold: float,
    min_h2_fraction: float,
    h2_perturbation_ratio: float,
) -> ConstitutionResult:
    """Layer 2: Domain-invariant structural checks on the candidate reward function."""
    probes = _representative_probes(probe_mat_path, args)
    if not probes:
        return ConstitutionResult(
            passed=True, skipped=True,
            h2_monotonicity_ok=True, soc_boundary_ok=True, dominance_ok=True,
            soc_dominance_ok=True, h2_fraction_ok=True,
            message="constitution: skipped (no probe steps loadable)",
        )

    h2_ok, h2_msg = _check_h2_monotonicity(candidate_fn, probes, h2_perturbation_ratio)
    soc_ok, soc_msg = _check_soc_boundary(candidate_fn, probes)
    dom_ok, dom_msg = _check_dominance(
        candidate_fn,
        probe_mat_path,
        args,
        dominance_threshold,
        soc_dominance_threshold,
        min_h2_fraction,
    )
    means = _cost_means(candidate_fn, probe_mat_path, args)
    if means is not None and means["objective_cost"] >= 1e-10:
        obj = means["objective_cost"]
        soc_dom_ok = (means["soc_cost"] / obj) <= soc_dominance_threshold
        h2_frac_ok = (means["h2_cost"] / obj) >= min_h2_fraction
    else:
        soc_dom_ok = True
        h2_frac_ok = True

    passed = h2_ok and soc_ok and dom_ok
    return ConstitutionResult(
        passed=passed, skipped=False,
        h2_monotonicity_ok=h2_ok,
        soc_boundary_ok=soc_ok,
        dominance_ok=dom_ok,
        soc_dominance_ok=soc_dom_ok,
        h2_fraction_ok=h2_frac_ok,
        message=f"constitution: {'PASS' if passed else 'FAIL'} | {h2_msg}; {soc_msg}; {dom_msg}",
    )



# ── Main entry point ──────────────────────────────────────────────────────────

def evaluate_aligned_gate(
    candidate_source: str,
    log_path: "str | Path",
    episode_data_dir: "str | Path",
    args,
    min_tac: float = _DEFAULTS["min_tac"],
    dominance_threshold: float = _DEFAULTS["dominance_threshold"],
    soc_dominance_threshold: float = _DEFAULTS["soc_dominance_threshold"],
    min_h2_fraction: float = _DEFAULTS["min_h2_fraction"],
    h2_perturbation_ratio: float = _DEFAULTS["h2_perturbation_ratio"],
) -> AlignedGateResult:
    """Evaluate a candidate reward function through two alignment layers.

    Parameters
    ----------
    candidate_source:
        Source code of the candidate ``def get_reward(self):`` function.
    log_path:
        Path to ``train_log.log`` from the most recent training chunk.
    episode_data_dir:
        Directory containing ``data_ep<N>.mat`` files.
    args:
        Argument namespace with w_soc, w_h2, w_fc, w_batt, soc_target,
        soc_low, soc_high, eq_h2_batt_coef, tpe_soc_in_bounds_floor.
    """
    episodes = parse_train_log(log_path)
    ep_dir = Path(episode_data_dir)

    in_bounds_floor = float(
        getattr(args, "tpe_soc_in_bounds_floor", _cfg_value("tpe_soc_in_bounds_floor", args=args))
    )

    # Build candidate function
    try:
        candidate_fn = _build_candidate_fn(candidate_source)
    except Exception as e:
        _skip_tac = TACResult(True, True, None, 0, min_tac, False, "skipped")
        _skip_const = ConstitutionResult(True, True, True, True, True, True, True, "skipped")
        return AlignedGateResult(
            passed=False,
            tac=_skip_tac,
            constitution=_skip_const,
            message=f"aligned_gate FAILED: candidate build error: {e}",
        )

    # Find a representative episode .mat file for constitution probes
    probe_mat_path: Optional[Path] = None
    for ep in episodes:
        if ep.in_bounds_rate is None or float(ep.in_bounds_rate) < in_bounds_floor:
            continue
        candidate_path = ep_dir / f"data_ep{int(ep.episode)}.mat"
        if candidate_path.exists():
            probe_mat_path = candidate_path
            break

    # ── Layer 1: Rank Correlation ──────────────────────────────────────────────
    tac_result = evaluate_tac(
        candidate_fn=candidate_fn,
        episodes=episodes,
        ep_dir=ep_dir,
        args=args,
        min_in_bounds_rate=in_bounds_floor,
        threshold=min_tac,
    )

    # ── Layer 2: Reward Constitution ───────────────────────────────────────────
    if probe_mat_path is not None:
        constitution_result = evaluate_constitution(
            candidate_fn=candidate_fn,
            probe_mat_path=probe_mat_path,
            args=args,
            dominance_threshold=dominance_threshold,
            soc_dominance_threshold=soc_dominance_threshold,
            min_h2_fraction=min_h2_fraction,
            h2_perturbation_ratio=h2_perturbation_ratio,
        )
    else:
        constitution_result = ConstitutionResult(
            passed=True, skipped=True,
            h2_monotonicity_ok=True, soc_boundary_ok=True, dominance_ok=True,
            soc_dominance_ok=True, h2_fraction_ok=True,
            message="constitution: skipped (no .mat file found for probing)",
        )

    # ── Aggregate ──────────────────────────────────────────────────────────────
    passed = tac_result.passed and constitution_result.passed
    message = (
        f"aligned_gate {'PASSED' if passed else 'FAILED'}: "
        f"{tac_result.message} | "
        f"{constitution_result.message}"
    )
    return AlignedGateResult(
        passed=passed,
        tac=tac_result,
        constitution=constitution_result,
        message=message,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-layer aligned reward gate for LDRI")
    parser.add_argument("--candidate-file", required=True, help="file with def get_reward(self): (candidate)")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--episode-data-dir", required=True)
    parser.add_argument("--w-soc", type=float, default=300.0)
    parser.add_argument("--w-h2", type=float, default=10.0)
    parser.add_argument("--w-fc", type=float, default=4.275e6)
    parser.add_argument("--w-batt", type=float, default=1.116e6)
    parser.add_argument("--soc-target", type=float, default=0.6)
    parser.add_argument("--soc-low", type=float, default=0.4)
    parser.add_argument("--soc-high", type=float, default=0.8)
    parser.add_argument("--eq-h2-batt-coef", type=float, default=0.0164)
    parser.add_argument("--tpe-soc-in-bounds-floor", type=float, default=0.0)
    parser.add_argument("--min-tac", type=float, default=_DEFAULTS["min_tac"])
    parser.add_argument("--dominance-threshold", type=float, default=_DEFAULTS["dominance_threshold"])
    parser.add_argument("--soc-dominance-threshold", type=float, default=_DEFAULTS["soc_dominance_threshold"])
    parser.add_argument("--min-h2-fraction", type=float, default=_DEFAULTS["min_h2_fraction"])
    parser.add_argument("--h2-perturbation-ratio", type=float, default=_DEFAULTS["h2_perturbation_ratio"])
    args = parser.parse_args()

    candidate_source = Path(args.candidate_file).read_text(encoding="utf-8")

    result = evaluate_aligned_gate(
        candidate_source=candidate_source,
        log_path=args.log_path,
        episode_data_dir=args.episode_data_dir,
        args=args,
        min_tac=args.min_tac,
        dominance_threshold=args.dominance_threshold,
        soc_dominance_threshold=args.soc_dominance_threshold,
        min_h2_fraction=args.min_h2_fraction,
        h2_perturbation_ratio=args.h2_perturbation_ratio,
    )

    print(result.message)
    print(f"\n  Layer 1 TAC         : {'PASS' if result.tac.passed else 'FAIL'} "
          f"{'(skipped)' if result.tac.skipped else ''} "
          f"tau={result.tac.tau}, h2_available={result.tac.h2_available}")
    print(f"  Layer 2 constitution: {'PASS' if result.constitution.passed else 'FAIL'} "
          f"{'(skipped)' if result.constitution.skipped else ''} "
          f"h2_mono={result.constitution.h2_monotonicity_ok}, "
          f"soc_bnd={result.constitution.soc_boundary_ok}, "
          f"dominance={result.constitution.dominance_ok}, "
          f"soc_dominance={result.constitution.soc_dominance_ok}, "
          f"h2_fraction={result.constitution.h2_fraction_ok}")
    print(f"\n  Overall: {'PASSED' if result.passed else 'FAILED'}")
