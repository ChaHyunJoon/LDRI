import ast
import hashlib
import json
import inspect
import re
from pathlib import Path
import subprocess
import datetime as dt
import textwrap
from typing import Any, Dict, List, Optional

from arguments_ldri import get_ldri_args

from feedback_parser import build_feedback_bundle
from prompt_builder import build_iteration_prompt, extract_method_source
from llm_client import LLMClient, get_api_key
from multi_agent_review import (
    run_multi_agent_review,
    build_review_packet,
    _compute_diagnostic_from_packet,
    _compute_scale_check_from_packet,
    _compute_trainability_from_packet,
    _check_counterfactual_risks,
    _extract_scale_coefficients,
)
from reward_patcher import (
    ensure_logging_contract,
    extract_method_from_llm_response,
    method_diff_preview,
    patch_method_in_file,
)
from ldri_observer import LDRIObserver, extract_analysis_section, infer_key_weight_changes
from reward_lr_tuner import auto_tune_lrs_for_reward_patch, ensure_lr_reference_attrs


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

RESUME_RESTORE_KEYS = [
    "DRL",
    "scenario_name",
    "MODE",
    "soc0",
    "soc_target",
    "w_soc",
    "w_h2",
    "w_fc",
    "w_batt",
    "eq_h2_batt_coef",
    "lr_critic",
    "lr_critic_min",
    "lr_actor",
    "lr_actor_min",
    "lr_alpha",
    "lr_schedule_episodes",
    "lr_restart_cycles",
    "lr_warmup_ratio",
    "auto_lr_on_reward_patch",
    "lr_auto_scale_floor",
    "lr_auto_scale_ceiling",
    "lr_auto_warmup_min",
    "lr_auto_warmup_max",
    "lr_auto_warmup_gain",
    "chunk_episodes",
    "feedback_process_episodes",
    "reset_agent_on_patch",
    "reset_episode_counter_on_patch",
    "reward_module",
    "reward_file",
    "prompt_template",
    "bootstrap_from_scratch",
    "bootstrap_only",
    "bootstrap_prompt_template",
    "bootstrap_retries",
    "bootstrap_task_description",
    "bootstrap_max_attrs",
    "bootstrap_max_methods",
    "llm_provider",
    "llm_model",
    "multi_agent_review",
    "parse_retries",
    "tpe_retries",
    "tpe_soc_in_bounds_floor",
    "soc_low",
    "soc_high",
    "aligned_min_tac",
    "aligned_dominance_threshold",
    "aligned_soc_dominance_threshold",
    "aligned_min_h2_fraction",
    "aligned_h2_perturbation_ratio",
]


def _fmt_tag_value(v):
    s = str(v)
    # Keep folder names shell-friendly and compact.
    return s.replace(" ", "").replace("/", "-").replace(".", "p")


def _llm_run_tag(args) -> str:
    provider = _fmt_tag_value(getattr(args, "llm_provider", "") or "llm")
    model = str(getattr(args, "llm_model", "") or "default")
    model_name = model.split("/")[-1]
    return f"{provider}_{_fmt_tag_value(model_name)}"


def _review_run_tag(args) -> str:
    if bool(getattr(args, "multi_agent_review", False)):
        return "_multiagent"
    return ""


def _create_run_root(args):
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{stamp}_{args.DRL}_{args.scenario_name}"
        f"_wsoc{_fmt_tag_value(args.w_soc)}"
        f"_iter{_fmt_tag_value(args.ldri_iterations)}"
        f"_chunk{_fmt_tag_value(args.chunk_episodes)}"
        f"_fb{_fmt_tag_value(args.feedback_process_episodes)}"
        f"_llm{_llm_run_tag(args)}"
        f"{_review_run_tag(args)}"
        f"_{args.file_v}"
    )
    run_root = Path(args.ldri_root).resolve() / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _configure_training_paths(args, run_root):
    # isolate LDRI workflow outputs from baseline directories
    args.save_dir = str((run_root / "models").resolve())
    args.log_dir = str((run_root / "tb_logs").resolve())
    args.eva_dir = str((run_root / "eval").resolve())


def _resolve_input_path(path_value: str) -> str:
    """Resolve CLI path robustly for both project-root and LDRI-root executions."""
    p = Path(path_value)
    ldri_dir = Path(__file__).resolve().parent
    alias_map = {
        "reward_bootstrap_codex_prompt.md": "reward_generation_prompt.md",
    }

    if p.is_absolute():
        if p.exists():
            return str(p.resolve())

        remapped_parts = ["LDRI" if part == "CARD" else part for part in p.parts]
        remapped_path = Path(*remapped_parts)
        if remapped_path.exists():
            return str(remapped_path.resolve())

        alias_name = alias_map.get(p.name, p.name)
        absolute_fallbacks = [
            p.parent / alias_name,
            remapped_path.parent / alias_name,
            ldri_dir / alias_name,
        ]
        for candidate in absolute_fallbacks:
            if candidate.exists():
                return str(candidate.resolve())
        return str(p)

    # 1) Relative to current working directory (explicit user intent)
    cwd_candidate = (Path.cwd() / p)
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())

    # 2) Relative to this LDRI module directory (default arguments)
    ldri_candidate = (ldri_dir / p)
    if ldri_candidate.exists():
        return str(ldri_candidate.resolve())

    # Backward-compatible aliases for pre-rename prompt filenames.
    alias_name = alias_map.get(p.name)
    if alias_name:
        alias_candidates = []
        if p.parent != Path("."):
            alias_candidates.append((Path.cwd() / p.parent / alias_name))
            alias_candidates.append((ldri_dir / p.parent / alias_name))
        else:
            alias_candidates.append(Path.cwd() / alias_name)
            alias_candidates.append(ldri_dir / alias_name)
        for alias_path in alias_candidates:
            if alias_path.exists():
                return str(alias_path.resolve())

    # 3) Fallback for clear downstream error message.
    return str(cwd_candidate.resolve())


def _init_llm_client(args):
    if args.skip_llm:
        return None
    api_key = get_api_key(args.llm_provider, args.llm_api_key, args.llm_api_key_env)
    return LLMClient(provider=args.llm_provider, model=args.llm_model, api_key=api_key)


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _register_reward_method(
    registry: Dict[str, Dict[str, Any]],
    method_source: str,
    origin: str,
    source_file: Optional[Path] = None,
) -> str:
    signature = _method_signature(method_source)
    entry = registry.get(signature)
    if entry is None:
        entry = {"signature": signature, "origins": [], "source_files": []}
        registry[signature] = entry

    if origin and origin not in entry["origins"]:
        entry["origins"].append(origin)

    if source_file is not None:
        source_text = str(source_file)
        if source_text not in entry["source_files"]:
            entry["source_files"].append(source_text)

    return signature


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


def _load_bootstrap_candidate_file(run_root: Path) -> Optional[Path]:
    boot_dir = run_root / "bootstrap"
    summary_path = boot_dir / "bootstrap_summary.json"
    if not summary_path.exists():
        return None

    try:
        payload = _read_json(summary_path)
    except Exception:
        return None

    if not bool(payload.get("patched")):
        return None

    selected_attempt = payload.get("selected_attempt")
    if selected_attempt is not None:
        try:
            candidate = boot_dir / f"candidate_get_reward_attempt_{int(selected_attempt)}.py"
        except Exception:
            candidate = None
        if candidate is not None and candidate.exists():
            return candidate

    attempt_logs = payload.get("attempt_logs", [])
    if not isinstance(attempt_logs, list):
        return None
    for record in attempt_logs:
        if not isinstance(record, dict) or not bool(record.get("smoke_ok")):
            continue
        candidate_file = record.get("candidate_file")
        if not candidate_file:
            continue
        candidate = Path(candidate_file)
        if candidate.exists():
            return candidate
    return None


def _load_iteration_candidate_file(iter_dir: Path) -> Optional[Path]:
    summary_path = iter_dir / "iteration_summary.json"
    if not summary_path.exists():
        return None

    try:
        payload = _read_json(summary_path)
    except Exception:
        return None

    outcome = payload.get("outcome", {})
    if not isinstance(outcome, dict) or not bool(outcome.get("reward_changed")):
        return None

    attempts = payload.get("attempts", [])
    if not isinstance(attempts, list):
        return None

    for record in attempts:
        if not isinstance(record, dict) or record.get("tpe_pass") is not True:
            continue
        candidate_file = record.get("candidate_file")
        if not candidate_file:
            continue
        candidate = Path(candidate_file)
        if candidate.exists():
            return candidate

    for record in attempts:
        if not isinstance(record, dict) or not bool(record.get("parse_ok")):
            continue
        if "tpe_disabled" not in str(record.get("note", "")):
            continue
        candidate_file = record.get("candidate_file")
        if not candidate_file:
            continue
        candidate = Path(candidate_file)
        if candidate.exists():
            return candidate

    return None


def _collect_trained_reward_history(
    run_root: Path,
    *,
    upto_iteration: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    history: Dict[str, Dict[str, Any]] = {}

    bootstrap_candidate = _load_bootstrap_candidate_file(run_root)
    if bootstrap_candidate is not None:
        try:
            _register_reward_method(
                history,
                bootstrap_candidate.read_text(encoding="utf-8"),
                origin="bootstrap",
                source_file=bootstrap_candidate,
            )
        except Exception:
            pass

    iter_dirs: List[tuple[int, Path]] = []
    for path in run_root.glob("iter_*"):
        if not path.is_dir():
            continue
        m = re.fullmatch(r"iter_(\d+)", path.name)
        if not m:
            continue
        iteration = int(m.group(1))
        if upto_iteration is not None and iteration >= int(upto_iteration):
            continue
        iter_dirs.append((iteration, path))

    for iteration, iter_dir in sorted(iter_dirs):
        candidate_file = _load_iteration_candidate_file(iter_dir)
        if candidate_file is None:
            continue
        try:
            _register_reward_method(
                history,
                candidate_file.read_text(encoding="utf-8"),
                origin=f"iter_{iteration:03d}",
                source_file=candidate_file,
            )
        except Exception:
            continue

    return history


def _build_historical_reward_guard(reward_history: Dict[str, Dict[str, Any]]) -> str:
    historical_origins: List[str] = []
    for entry in reward_history.values():
        origins = entry.get("origins", [])
        if not isinstance(origins, list):
            continue
        for origin in origins:
            origin_text = str(origin)
            if origin_text.startswith("current_iter_"):
                continue
            historical_origins.append(origin_text)

    seen = set()
    ordered_origins = []
    for origin in sorted(historical_origins, key=_origin_sort_key):
        if origin in seen:
            continue
        seen.add(origin)
        ordered_origins.append(origin)

    lines = [
        "### Historical Reward Guard",
        "- Returning the current reward unchanged is invalid because it has already been trained.",
        "- Returning any previously trained reward variant is invalid even if it would improve the instantaneous TPE delta.",
        "- Exact structural matches to already-trained `get_reward(self)` variants are rejected before TPE evaluation.",
    ]
    if ordered_origins:
        lines.append("- Historical reward versions already used in this run:")
        shown = ordered_origins[:8]
        for origin in shown:
            lines.append(f"  - {origin}")
        if len(ordered_origins) > len(shown):
            lines.append(f"  - ... {len(ordered_origins) - len(shown)} more omitted")

    return "\n".join(lines).strip()


def _extract_iteration_numbers(summary: Dict[str, Any]):
    iterations = []
    for item in summary.get("iterations", []):
        if not isinstance(item, dict):
            continue
        try:
            iterations.append(int(item.get("iteration")))
        except Exception:
            continue
    return sorted(set(iterations))


def _resolve_run_root(args):
    resume_run_root = str(getattr(args, "resume_run_root", "") or "").strip()
    if resume_run_root:
        run_root = Path(resume_run_root).resolve()
        if not run_root.exists() or not run_root.is_dir():
            raise RuntimeError(f"invalid --resume_run_root: {run_root}")
        return run_root, True
    return _create_run_root(args), False


def _apply_resume_run_config(args, run_config: Dict[str, Any]):
    if not isinstance(run_config, dict):
        raise RuntimeError("invalid run_config.json: expected a JSON object")

    overridden = []
    for key in RESUME_RESTORE_KEYS:
        if key not in run_config:
            continue
        if not hasattr(args, key):
            continue
        saved_value = run_config.get(key)
        if saved_value is None:
            continue
        current_value = getattr(args, key)
        if current_value != saved_value:
            setattr(args, key, saved_value)
            overridden.append((key, current_value, saved_value))
    return overridden


def _build_prompt(args, feedback_bundle, current_method, iteration, reward_history=None):
    replacements = {
        "RL_ALGO": args.DRL,
        "DRIVING_CYCLES": args.scenario_name,
        "EPISODE_STEPS": str(args.episode_steps),
        "REWARD_VERSION": f"ldri_iter_{iteration}",
        "TPE_SOC_FLOOR": f"{float(args.tpe_soc_in_bounds_floor):.1f}",
        "CURRENT_GET_REWARD_CODE": current_method.strip("\n"),
        "PROCESS_FEEDBACK": feedback_bundle["PROCESS_FEEDBACK"],
        "TRAJECTORY_FEEDBACK": feedback_bundle["TRAJECTORY_FEEDBACK"],
        "PREFERENCE_FEEDBACK": feedback_bundle["PREFERENCE_FEEDBACK"],
    }
    prompt = build_iteration_prompt(args.prompt_template, replacements)
    history_block = _build_historical_reward_guard(reward_history or {})
    if history_block:
        prompt = prompt.rstrip() + "\n\n" + history_block + "\n"
    return prompt


def _maybe_auto_tune_lr_after_patch(
    *,
    args,
    observer: LDRIObserver,
    iter_info: Dict[str, Any],
    current_method: str,
    candidate_method: str,
    chunk_result,
    lr_proposal: Optional[Dict[str, Any]] = None,
) -> None:
    tuning = auto_tune_lrs_for_reward_patch(
        current_method_source=current_method,
        candidate_method_source=candidate_method,
        log_path=chunk_result.log_path,
        episode_data_dir=chunk_result.episode_data_dir,
        args=args,
        lr_proposal=lr_proposal,
    )
    iter_info["auto_lr"] = tuning
    if not bool(tuning.get("applied")):
        observer.status("LR_TUNE_SKIPPED", str(tuning.get("reason", "auto LR tuning unavailable")))
        return

    applied = tuning["applied_settings"]
    observer.status(
        "LR_TUNED",
        (
            f"factor={float(tuning['applied_factor']):.3f}, "
            f"critic={float(applied['lr_critic']):.6f}/{float(applied['lr_critic_min']):.6f}, "
            f"actor={float(applied['lr_actor']):.6f}/{float(applied['lr_actor_min']):.6f}, "
            f"warmup_ratio={float(applied['lr_warmup_ratio']):.3f}, "
            f"scale_ratio={float(tuning['scale_ratio']):.3f}, "
            f"rms_ratio={float(tuning['rms_ratio']):.3f}, "
            f"trend_factor={float(tuning['trend_summary']['trend_factor']):.3f}"
        ),
    )


def _truncate_repr(value: Any, max_len: int = 120) -> str:
    try:
        txt = repr(value)
    except Exception:
        txt = f"<{type(value).__name__}>"
    txt = " ".join(txt.split())
    if len(txt) > max_len:
        return txt[: max_len - 3] + "..."
    return txt


def _safe_signature(fn) -> str:
    try:
        return str(inspect.signature(fn))
    except Exception:
        return "(...)"


def _build_environment_description(args, env) -> str:
    agent = env.agent
    attr_items = sorted(
        (
            (name, value)
            for name, value in vars(agent).items()
            if not name.startswith("_") and not callable(value)
        ),
        key=lambda x: x[0],
    )
    method_items = [
        (name, fn)
        for name, fn in inspect.getmembers(agent.__class__, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    method_items.sort(key=lambda x: x[0])

    lines = [
        "# Environment abstraction (Pythonic summary for reward design)",
        "import numpy as np",
        "",
        f"class {env.__class__.__name__}:",
        "    def reset(self) -> np.ndarray: ...",
        "    def step(self, action, episode_step: int) -> tuple[np.ndarray, float, bool, dict]: ...",
        "",
        f"class {agent.__class__.__name__}:",
        "    # inheritance: " + " <- ".join(cls.__name__ for cls in agent.__class__.mro()),
        "",
        "    # scalar/state attributes accessible in get_reward(self)",
    ]

    max_attrs = max(1, int(getattr(args, "bootstrap_max_attrs", 80)))
    for name, value in attr_items[:max_attrs]:
        lines.append(f"    self.{name}: {type(value).__name__} = {_truncate_repr(value)}")
    if len(attr_items) > max_attrs:
        lines.append(f"    # ... {len(attr_items) - max_attrs} more attributes omitted")

    lines.append("")
    lines.append("    # callable methods on EMS")
    max_methods = max(1, int(getattr(args, "bootstrap_max_methods", 40)))
    for name, fn in method_items[:max_methods]:
        lines.append(f"    def {name}{_safe_signature(fn)}: ...")
    if len(method_items) > max_methods:
        lines.append(f"    # ... {len(method_items) - max_methods} more methods omitted")

    simple_types = (int, float, bool, str, bytes, tuple, list, dict, set, type(None))
    composed = []
    for name, value in attr_items:
        if isinstance(value, simple_types):
            continue
        if value.__class__.__module__.startswith("numpy"):
            continue
        cls = value.__class__
        if cls.__module__ == "builtins":
            continue
        composed.append((name, value))

    if composed:
        lines.append("")
        lines.append("    # composed objects and their representative callables")
        for name, value in composed[:8]:
            cls = value.__class__
            lines.append(f"    self.{name}: {cls.__module__}.{cls.__name__}")
            sub_methods = [
                m
                for m, fn in inspect.getmembers(cls, predicate=inspect.isfunction)
                if not m.startswith("_")
            ]
            if sub_methods:
                shown = ", ".join(sub_methods[:8])
                lines.append(f"    # callable: {shown}")

    lines.extend(
        [
            "",
            "# Runtime context",
            f"DRL = {args.DRL!r}",
            f"DRIVING_CYCLE = {args.scenario_name!r}",
            f"EPISODE_STEPS = {int(args.episode_steps)}",
            f"SOC_BOUNDS = ({float(args.soc_low):.4f}, {float(args.soc_high):.4f})",
            f"SOC_TARGET = {float(args.soc_target):.4f}",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _build_bootstrap_task_description(args) -> str:
    base = textwrap.dedent(
        f"""
        Design `EMS.get_reward(self)` for fuel-cell hybrid EMS control from scratch.
        - Primary objective: minimize per-step objective cost (fuel-cell H2 use + FC degradation + battery degradation + SOC regulation penalties).
        - Treat `h2_cost` as pure fuel-cell hydrogen consumption based on `h2_fcs`, not equivalent-hydrogen battery credit/debit.
        - Safety objective: keep SOC inside [{float(args.soc_low):.3f}, {float(args.soc_high):.3f}] and near target {float(args.soc_target):.3f}.
        - Behavior objective: avoid unstable control oscillations and reward hacking.
        - Mandatory weighted skeleton:
          `soc_cost` anchored on `float(self.w_soc) * float(soc - soc_target) ** 2`,
          `h2_cost = float(self.w_h2) * float(self.h2_fcs)`,
          `fcs_soh_cost = float(self.w_fc) * float(self.dSOH_FCS)`,
          `batt_soh_cost = float(self.w_batt) * float(self.dSOH_batt)`.
        - Logging contract: `self.info` must contain per-step sub-reward terms and final scalar reward.
        - Log `float(self.h2_batt)` and `float(self.h2_equal)` in `self.info` for logging only; do not read them in reward computation logic.
        - Implementation constraints: use only attributes available on `self` from environment abstraction.
        """
    ).strip()

    extra = str(getattr(args, "bootstrap_task_description", "") or "").strip()
    if extra:
        base = base + "\n\nAdditional task guidance:\n" + extra
    return base + "\n"


def _build_reward_template_contract() -> str:
    return textwrap.dedent(
        """
        def get_reward(self):
            # 1) Read core states and constraints
            soc = float(self.SOC)
            soc_low, soc_high = self.soc_bounds
            soc_target = self.SOC_target

            # 2) Build hydrogen terms first
            h2_fcs = float(self.h2_fcs)
            # 3) Build non-negative weighted sub-costs
            soc_cost = float(self.w_soc) * float(soc - soc_target) ** 2
            # Optional: add stronger quadratic out-of-bounds strengthening here,
            # but keep it inside soc_cost and keep it weighted by self.w_soc.
            h2_cost = float(self.w_h2) * float(h2_fcs)
            fcs_soh_cost = float(self.w_fc) * float(self.dSOH_FCS)
            batt_soh_cost = float(self.w_batt) * float(self.dSOH_batt)
            objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost

            # 4) Reward is scalar float
            reward = float(-objective_cost)

            # 5) Update required per-step components for logging and analysis
            self.info.update({
                "EMS_reward": reward,
                "h2_fcs": float(h2_fcs),
                "h2_batt": float(self.h2_batt),
                "h2_equal": float(self.h2_equal),
                "soc_cost": float(soc_cost),
                "h2_cost": float(h2_cost),
                "fcs_soh_cost": float(fcs_soh_cost),
                "batt_soh_cost": float(batt_soh_cost),
                "objective_cost": float(objective_cost),
                "soc_in_bounds": int(soc_low < soc < soc_high),
            })
            return reward
        """
    ).strip()


def _build_bootstrap_prompt(args, env_description: str, task_description: str) -> str:
    replacements = {
        "RL_ALGO": args.DRL,
        "DRIVING_CYCLES": args.scenario_name,
        "EPISODE_STEPS": str(args.episode_steps),
        "REWARD_VERSION": "bootstrap_init",
        "TPE_SOC_FLOOR": f"{float(args.tpe_soc_in_bounds_floor):.1f}",
        "ENVIRONMENT_DESCRIPTION": env_description.strip("\n"),
        "TASK_DESCRIPTION": task_description.strip("\n"),
        "REWARD_TEMPLATE": _build_reward_template_contract(),
        "REQUIRED_INFO_KEYS": ", ".join(REQUIRED_INFO_KEYS),
    }
    return build_iteration_prompt(args.bootstrap_prompt_template, replacements)


def _smoke_test_candidate_method(method_source: str, env) -> Dict[str, Any]:
    import numpy as np

    namespace = {"np": np}
    exec(textwrap.dedent(method_source), namespace, namespace)
    fn = namespace.get("get_reward")
    if fn is None or not callable(fn):
        raise RuntimeError("candidate does not define callable get_reward(self)")

    if len(env.speed_list) == 0 or len(env.acc_list) == 0:
        raise RuntimeError("driving cycle is empty; cannot run candidate smoke test")

    agent = env.agent
    agent.reset_obs()
    action = np.array([0.0], dtype=np.float32)
    agent.execute(action, float(env.speed_list[0]), float(env.acc_list[0]))
    agent.info = {}

    reward_val = fn(agent)
    if isinstance(reward_val, tuple):
        reward_val = reward_val[0]
    reward = float(reward_val)
    if not np.isfinite(reward):
        raise RuntimeError(f"candidate reward is non-finite: {reward}")

    missing_keys = [k for k in REQUIRED_INFO_KEYS if k not in agent.info]
    if missing_keys:
        raise RuntimeError(f"candidate missing required info keys: {missing_keys}")

    for key in REQUIRED_INFO_KEYS:
        val = agent.info.get(key)
        if key == "soc_in_bounds":
            int(val)
            continue
        num = float(val)
        if not np.isfinite(num):
            raise RuntimeError(f"candidate info key '{key}' is non-finite: {num}")

    return {
        "smoke_reward": reward,
        "info_snapshot": {k: agent.info.get(k) for k in REQUIRED_INFO_KEYS},
    }


def _run_bootstrap_from_scratch(args, client, env, run_root: Path) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("bootstrap_from_scratch requires initialized LLM client")

    boot_dir = run_root / "bootstrap"
    boot_dir.mkdir(parents=True, exist_ok=True)

    env_description = _build_environment_description(args, env)
    task_description = _build_bootstrap_task_description(args)
    prompt = _build_bootstrap_prompt(args, env_description, task_description)

    _write_text(boot_dir / "environment_description.py.txt", env_description)
    _write_text(boot_dir / "task_description.txt", task_description)
    _write_text(boot_dir / "reward_template_contract.py", _build_reward_template_contract() + "\n")
    _write_text(boot_dir / "prompt.md", prompt)

    attempts = 1 + max(0, int(getattr(args, "bootstrap_retries", 0)))
    try:
        current_method = extract_method_source(args.reward_file, class_name="EMS", method_name="get_reward")
    except Exception:
        current_method = (
            "def get_reward(self):\n"
            "    # no existing method in target file (bootstrap insert mode)\n"
            "    return 0.0\n"
        )
    parse_ok_count = 0
    selected_method = None
    selected_attempt = None
    smoke_result = None
    attempt_logs = []

    print(f"[bootstrap] generating initial reward from scratch (attempts={attempts})")
    for attempt in range(1, attempts + 1):
        response = client.complete(
            prompt=prompt,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout_sec=args.llm_timeout_sec,
        )
        response_file = boot_dir / f"llm_response_attempt_{attempt}.md"
        _write_text(response_file, response)
        _write_text(boot_dir / f"analysis_attempt_{attempt}.md", extract_analysis_section(response))

        rec: Dict[str, Any] = {
            "attempt": attempt,
            "response_file": str(response_file),
            "parse_ok": False,
            "smoke_ok": False,
            "note": "",
        }

        try:
            candidate_method = extract_method_from_llm_response(response, method_name="get_reward")
        except Exception as e:
            rec["note"] = f"parse_fail: {e}"
            attempt_logs.append(rec)
            continue
        candidate_method = ensure_logging_contract(candidate_method, REQUIRED_INFO_KEYS)

        parse_ok_count += 1
        rec["parse_ok"] = True
        cand_file = boot_dir / f"candidate_get_reward_attempt_{attempt}.py"
        _write_text(cand_file, candidate_method)
        rec["candidate_file"] = str(cand_file)

        diff_text = method_diff_preview(current_method, candidate_method)
        _write_text(boot_dir / f"candidate_vs_current_attempt_{attempt}.diff", diff_text)

        try:
            smoke = _smoke_test_candidate_method(candidate_method, env)
            _write_json(boot_dir / f"smoke_test_attempt_{attempt}.json", smoke)
        except Exception as e:
            rec["note"] = f"smoke_fail: {e}"
            attempt_logs.append(rec)
            continue

        rec["smoke_ok"] = True
        rec["note"] = "ready"
        attempt_logs.append(rec)
        selected_method = candidate_method
        selected_attempt = attempt
        smoke_result = smoke
        break

    result: Dict[str, Any] = {
        "enabled": True,
        "attempts": attempts,
        "parse_ok_count": parse_ok_count,
        "selected_attempt": selected_attempt,
        "patched": False,
        "attempt_logs": attempt_logs,
    }

    if selected_method is None:
        result["status"] = "bootstrap_failed"
        result["message"] = "all bootstrap attempts failed parse/smoke checks"
        _write_json(boot_dir / "bootstrap_summary.json", result)
        return result

    result["status"] = "candidate_validated"
    result["smoke_result"] = smoke_result

    if args.skip_patch:
        result["status"] = "skip_patch"
        _write_json(boot_dir / "bootstrap_summary.json", result)
        return result

    patch_result = patch_method_in_file(
        file_path=args.reward_file,
        method_source=selected_method,
        class_name="EMS",
        method_name="get_reward",
        required_info_keys=REQUIRED_INFO_KEYS,
        backup=True,
        insert_if_missing=True,
    )
    subprocess.run(["python3", "-m", "py_compile", args.reward_file], check=True)

    result.update(
        {
            "status": "patched",
            "patched": True,
            "patch_target": patch_result.target_file,
            "backup_file": patch_result.backup_file,
        }
    )
    _write_json(boot_dir / "bootstrap_summary.json", result)
    return result


def _delta(pref: Optional[float], disp: Optional[float]) -> Optional[float]:
    if pref is None or disp is None:
        return None
    return float(pref) - float(disp)


def _better_delta(new_val: Optional[float], old_val: Optional[float]) -> bool:
    if new_val is None:
        return False
    if old_val is None:
        return True
    return new_val > old_val

def _build_aligned_gate_feedback(aligned) -> str:
    """Build LLM retry prompt section from a failed AlignedGateResult."""
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
# Single-agent guardrail helpers (mirror Python rules from multi-agent pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _build_diagnostic_hint_block(diagnostic_result: Dict[str, Any]) -> str:
    """Inject Python-computed diagnostic summary into the LLM prompt."""
    lines = ["\n\n# Pre-Refinement Diagnostic (Python-computed — treat as ground truth)"]
    lines.append(f"- Failure mode: {diagnostic_result.get('failure_mode', 'unknown')}")
    lines.append(f"- Convergence quality: {diagnostic_result.get('convergence_quality', 'unknown')}")
    lines.append(f"- H2 trend: {diagnostic_result.get('h2_trend_diagnosis', 'unknown')}")
    lines.append(f"- SOC stability: {diagnostic_result.get('soc_stability_diagnosis', 'unknown')}")
    oob = diagnostic_result.get("oob_rate", 0)
    lines.append(f"- OOB rate: {oob:.1%}")
    swing = diagnostic_result.get("mean_soc_swing")
    lines.append(f"- Mean SOC swing: {swing:.3f}" if swing is not None else "- Mean SOC swing: N/A")
    hints = diagnostic_result.get("reviewer_hints", {})
    fuel_hint = hints.get("fuel_efficiency", "")
    battery_hint = hints.get("battery_usage", "")
    if fuel_hint:
        lines.append(f"\nFuel efficiency: {fuel_hint}")
    if battery_hint:
        lines.append(f"Battery usage: {battery_hint}")
    lines.append(
        "\nUse this diagnostic to decide which reward term to modify. "
        "Do NOT ignore the failure mode classification above."
    )
    return "\n".join(lines)


def _build_scale_feedback_block(scale_result: Dict[str, Any]) -> str:
    """Convert scale_result violations into a mandatory prompt feedback block."""
    lines = ["\n\n# Coefficient Scale Violations (Python-enforced — MUST fix before submitting)"]
    for v in scale_result.get("violations", []):
        lines.append(
            f"- {v['coeff']}: current={v.get('current')} violates {v.get('cap_violated')} "
            f"[severity={v.get('severity')}]. Prescribed value: {v.get('prescribed')}."
        )
    hard = [v for v in scale_result.get("violations", []) if v.get("severity") == "hard_reject"]
    if hard:
        lines.append(
            "ALL hard_reject violations above MUST be corrected to their prescribed values. "
            "The candidate will be rejected again until every hard cap is satisfied."
        )
    else:
        lines.append("Address all soft violations to avoid degraded training stability.")
    return "\n".join(lines)


def _make_orchestrator_spec_from_changes(
    current_method: str, candidate_method: str
) -> Dict[str, Any]:
    """Synthesize a minimal orchestrator_spec from coefficient deltas in the code diff."""
    try:
        curr_coeffs = _extract_scale_coefficients(candidate_method)
        prev_coeffs = _extract_scale_coefficients(current_method)
    except Exception:
        return {"reward_spec": {"must_change": []}}

    must_change = []
    for name, curr_val in curr_coeffs.items():
        if curr_val is None:
            continue
        prev_val = prev_coeffs.get(name)
        if prev_val is None:
            must_change.append({"target": name, "direction": "add"})
        elif curr_val > prev_val * 1.01:
            must_change.append({"target": name, "direction": "increase"})
        elif curr_val < prev_val * 0.99:
            must_change.append({"target": name, "direction": "decrease"})

    soc_terms_curr = len(re.findall(r"\bsoc_cost\b.*?\+", candidate_method))
    soc_terms_prev = len(re.findall(r"\bsoc_cost\b.*?\+", current_method))
    if soc_terms_curr > soc_terms_prev:
        must_change.append({"target": "soc_term", "direction": "add"})

    w_h2_curr = re.search(r"\bw_h2\b.*?=.*?([\d.]+)", candidate_method)
    w_h2_prev = re.search(r"\bw_h2\b.*?=.*?([\d.]+)", current_method)
    if w_h2_curr and w_h2_prev:
        try:
            if float(w_h2_curr.group(1)) < float(w_h2_prev.group(1)) * 0.99:
                must_change.append({"target": "w_h2", "direction": "decrease"})
        except ValueError:
            pass

    return {"reward_spec": {"must_change": must_change}}


def _make_synthetic_reviewer_outputs(
    review_packet: Dict[str, Any], diagnostic_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Synthesize minimal reviewer_outputs for the counterfactual evaluator."""
    total_obj = 0.0
    total_h2 = 0.0
    count = 0
    for ep in review_packet.get("recent_episodes", []):
        obj = ep.get("mean_objective_cost")
        h2c = ep.get("mean_h2_cost")
        if obj is not None and h2c is not None and obj > 0:
            total_obj += obj
            total_h2 += h2c
            count += 1
    h2_fraction = (total_h2 / total_obj) if (count > 0 and total_obj > 0) else 0.5

    mean_swing = diagnostic_result.get("mean_soc_swing")
    battery_buffering = mean_swing is None or mean_swing >= 0.05

    return {
        "fuel_efficiency": {"h2_cost_fraction": h2_fraction},
        "battery_usage": {"battery_buffering_active": battery_buffering},
        "soc_safety": {"over_shaping": False},
    }


# input: prompt.md 파일의 내용 & 직전 학습 로그, 현재 reward source
def _run_llm_with_tpe_retry(
    args,
    client,
    prompt: str,
    iter_dir: Path,
    chunk_result,
    observer: LDRIObserver,
    current_method: str,
    reward_history: Optional[Dict[str, Dict[str, Any]]] = None,
    feedback_bundle: Optional[Dict[str, str]] = None,
    iteration: Optional[int] = None,
    prev_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate candidate method(s), log every attempt, and gate by aligned gate."""
    from ldri_aligned_gate import evaluate_aligned_gate

    max_parse_attempts = 1 + max(0, int(getattr(args, "parse_retries", args.tpe_retries)))
    max_tpe_attempts = 1 + max(0, int(args.tpe_retries))
    current_prompt = prompt

    parse_ok_count = 0
    parse_fail_count = 0
    historical_duplicate_count = 0
    tpe_eval_count = 0
    best_tpe_delta = None
    best_candidate_method = None
    best_candidate_attempt = None

    passed_candidate_method = None
    passed_tpe = None

    # ── Python-based guardrails (shared with multi-agent workflow) ─────────────
    _guardrails_available = False
    _diagnostic_result: Optional[Dict[str, Any]] = None
    _trainability_result: Optional[Dict[str, Any]] = None
    _base_review_packet: Optional[Dict[str, Any]] = None

    if feedback_bundle is not None:
        try:
            _base_review_packet = build_review_packet(
                feedback_bundle=feedback_bundle,
                current_method=current_method,
                chunk_result=chunk_result,
                args=args,
                iteration=iteration if iteration is not None else observer.iteration,
            )
            _diagnostic_result = _compute_diagnostic_from_packet(_base_review_packet)
            _trainability_result = _compute_trainability_from_packet(_base_review_packet)
            _guardrails_available = True
            _write_json(iter_dir / "diagnostic_result.json", _diagnostic_result)
            _write_json(iter_dir / "trainability_result.json", _trainability_result)
            _diag_block = _build_diagnostic_hint_block(_diagnostic_result)
            if _diag_block:
                current_prompt = current_prompt + _diag_block
                _write_text(iter_dir / "prompt_with_diagnostic.md", current_prompt)
        except Exception as _gex:
            print(f"[guardrails] initialization failed (non-fatal): {_gex}")
    # ──────────────────────────────────────────────────────────────────────────

    attempt = 1
    while attempt <= max_parse_attempts and tpe_eval_count < max_tpe_attempts:
        observer.status(f"CANDIDATE_{attempt}", "requesting LLM candidate")
        
        # LLM의 response를 llm_response_attempt_#.md 파일에 저장
        response = client.complete(
            prompt=current_prompt,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout_sec=args.llm_timeout_sec,
        )
        response_file = iter_dir / f"llm_response_attempt_{attempt}.md"
        _write_text(response_file, response)

        # llm_response_attempt_#.md의 analysis 부분만 잘라 analysis_response_attempt_#.md 파일에 저장
        analysis = extract_analysis_section(response)
        analysis_file = iter_dir / f"analysis_attempt_{attempt}.md"
        _write_text(analysis_file, analysis)

        attempt_rec: Dict[str, Any] = {
            "attempt": attempt,
            "response_file": str(response_file),
            "analysis_file": str(analysis_file),
            "parse_ok": False,
            "tpe_pass": None,
            "preferred_avg": None,
            "dispreferred_avg": None,
            "margin": None,
            "delta": None,
            "note": "",
        }
        # llm response에서 get_reward function만 추출한 뒤, 
        # candidate_get_reward_attempt_#.py로 refine이후 reward 후보군이 저장됨
        try:
            candidate_method = extract_method_from_llm_response(response, method_name="get_reward")
        # parse fail 시 실패 이유를 prompt_attempt_#+1.md에 추가
        except Exception as e:
            parse_fail_count += 1
            attempt_rec["note"] = f"parse_fail: {e}"
            observer.add_attempt(attempt_rec)
            if attempt < max_parse_attempts and tpe_eval_count < max_tpe_attempts:
                current_prompt = (
                    current_prompt
                    + "\n\n# Parse Feedback (must follow exactly)\n"
                    + f"Previous response parse failed: {e}\n"
                    + "Output MUST include a valid Python definition for `def get_reward(self):`.\n"
                    + "Prefer returning only one Python fenced code block containing the full method body.\n"
                    + "Do not include placeholders, pseudocode, or omitted sections."
                )
                _write_text(iter_dir / f"prompt_attempt_{attempt+1}.md", current_prompt)
            attempt += 1
            continue
        candidate_method = ensure_logging_contract(candidate_method, REQUIRED_INFO_KEYS)

        parse_ok_count += 1
        attempt_rec["parse_ok"] = True
        candidate_file = iter_dir / f"candidate_get_reward_attempt_{attempt}.py"
        _write_text(candidate_file, candidate_method)
        attempt_rec["candidate_file"] = str(candidate_file)

        candidate_signature = _method_signature(candidate_method)
        attempt_rec["candidate_signature"] = candidate_signature

        # key_weight_changes_attempt_#.json에 기존 reward와 현재의 reward 사이에 변화한 weight, penalty, scale을 기록 
        key_changes = infer_key_weight_changes(current_method, candidate_method)
        key_change_file = iter_dir / f"key_weight_changes_attempt_{attempt}.json"
        _write_json(key_change_file, {"changes": key_changes})
        attempt_rec["key_weight_changes_file"] = str(key_change_file)
        attempt_rec["key_weight_changes"] = key_changes

        history_entry = (reward_history or {}).get(candidate_signature)
        if history_entry is not None:
            historical_duplicate_count += 1
            matched_label = _history_match_label(history_entry)
            attempt_rec["historical_duplicate"] = True
            attempt_rec["historical_match"] = matched_label
            attempt_rec["note"] = f"historical_duplicate: matches {matched_label}"
            observer.status(
                "HISTORICAL_DUPLICATE",
                f"attempt {attempt}: candidate matches trained reward {matched_label}",
            )
            observer.add_attempt(attempt_rec)
            if attempt < max_parse_attempts and tpe_eval_count < max_tpe_attempts:
                current_prompt = (
                    current_prompt
                    + "\n\n# Historical Reward Rejection (must address)\n"
                    + f"Previous candidate structurally matched an already-trained reward: {matched_label}.\n"
                    + "Do not return the current reward unchanged.\n"
                    + "Do not revert to any previously trained reward variant.\n"
                    + "Propose a genuinely new local refinement to get_reward(self).\n"
                )
                _write_text(iter_dir / f"prompt_attempt_{attempt+1}.md", current_prompt)
            attempt += 1
            continue

        # ── Scale guardrail (block before aligned gate if coefficients violate caps) ──
        if _guardrails_available and _base_review_packet is not None:
            try:
                _cand_packet = dict(_base_review_packet)
                _cand_packet["reward_source"] = candidate_method
                _scale_result = _compute_scale_check_from_packet(
                    _cand_packet, prev_source=(prev_method or current_method)
                )
                _write_json(
                    iter_dir / f"scale_check_attempt_{attempt}.json", _scale_result
                )
                if _scale_result["decision"] == "hard_reject":
                    attempt_rec["note"] = (
                        "scale_hard_reject: "
                        + "; ".join(
                            f"{v['coeff']}={v.get('current')} "
                            f"violates {v.get('cap_violated')}"
                            for v in _scale_result["violations"]
                        )
                    )
                    observer.status("SCALE_HARD_REJECT", attempt_rec["note"])
                    observer.add_attempt(attempt_rec)
                    if attempt < max_parse_attempts and tpe_eval_count < max_tpe_attempts:
                        current_prompt = (
                            current_prompt + _build_scale_feedback_block(_scale_result)
                        )
                        _write_text(
                            iter_dir / f"prompt_attempt_{attempt + 1}.md", current_prompt
                        )
                    attempt += 1
                    continue
                elif _scale_result["decision"] == "soft_revise":
                    # Warn but allow this attempt through; feedback shapes next generation
                    if attempt < max_parse_attempts and tpe_eval_count < max_tpe_attempts:
                        current_prompt = (
                            current_prompt + _build_scale_feedback_block(_scale_result)
                        )
            except Exception as _sex:
                print(f"[guardrails] scale check failed (non-fatal): {_sex}")
        # ──────────────────────────────────────────────────────────────────────

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
        delta = float(aligned.tac.tau) if aligned.tac.tau is not None else None
        attempt_rec["tpe_pass"] = bool(aligned.passed)
        attempt_rec["preferred_avg"] = None
        attempt_rec["dispreferred_avg"] = None
        attempt_rec["margin"] = None
        attempt_rec["delta"] = delta
        attempt_rec["note"] = aligned.message
        if _better_delta(delta, best_tpe_delta):
            best_tpe_delta = delta
            best_candidate_method = candidate_method
            best_candidate_attempt = attempt
        elif best_candidate_method is None:
            best_candidate_method = candidate_method
            best_candidate_attempt = attempt
        status = "GATE_PASS" if aligned.passed else "GATE_FAIL"
        detail = f"attempt {attempt}: {aligned.message}"
        observer.status(status, detail)
        print(f"[iter {observer.iteration:03d}] {detail} -> {'PASS' if aligned.passed else 'FAIL'}")
        observer.add_attempt(attempt_rec)
        if aligned.passed:
            passed_candidate_method = candidate_method
            passed_tpe = aligned
            break
        if attempt < max_parse_attempts and tpe_eval_count < max_tpe_attempts:
            _retry_prompt = current_prompt + _build_aligned_gate_feedback(aligned)
            # Counterfactual risk warnings (Python-computed, appended after gate feedback)
            if (
                _guardrails_available
                and _diagnostic_result is not None
                and _base_review_packet is not None
            ):
                try:
                    _orch_spec = _make_orchestrator_spec_from_changes(
                        current_method, candidate_method
                    )
                    _synth_rev = _make_synthetic_reviewer_outputs(
                        _base_review_packet, _diagnostic_result
                    )
                    _synth_med = {
                        "blocked_changes": [],
                        "h2_suppression_risk_active": (
                            _diagnostic_result.get("failure_mode") == "h2_suppression"
                        ),
                    }
                    _cf_result = _check_counterfactual_risks(
                        _orch_spec, _diagnostic_result, _synth_rev, _synth_med
                    )
                    _write_json(
                        iter_dir / f"counterfactual_check_attempt_{attempt}.json",
                        _cf_result,
                    )
                    if _cf_result.get("warning_lines"):
                        _retry_prompt += (
                            "\n\n# Counterfactual Risk Warnings (Python-computed)\n"
                            + "\n".join(_cf_result["warning_lines"])
                        )
                except Exception as _cfex:
                    print(f"[guardrails] counterfactual check failed (non-fatal): {_cfex}")
            current_prompt = _retry_prompt
            _write_text(iter_dir / f"prompt_attempt_{attempt+1}.md", current_prompt)

        attempt += 1

    return {
        "parse_ok_count": parse_ok_count,
        "parse_fail_count": parse_fail_count,
        "historical_duplicate_count": historical_duplicate_count,
        "trainability_lr_proposal": (
            _trainability_result.get("lr_proposal") if _trainability_result is not None else None
        ),
        "tpe_eval_count": tpe_eval_count,
        "max_parse_attempts": max_parse_attempts,
        "max_tpe_attempts": max_tpe_attempts,
        "best_tpe_delta": best_tpe_delta,
        "best_candidate_method": best_candidate_method,
        "best_candidate_attempt": best_candidate_attempt,
        "passed_candidate_method": passed_candidate_method,
        "passed_tpe": passed_tpe,
    }


def _finalize_iteration(
    *,
    observer: LDRIObserver,
    iter_info: Dict[str, Any],
    reason: str,
    reward_changed: bool,
    source_file: str,
    backup_file: Optional[str],
) -> None:
    iter_info["final_reason"] = reason
    iter_info["reward_changed"] = bool(reward_changed)
    iter_info["source_file"] = source_file
    iter_info["backup_file"] = backup_file
    observer.set_outcome(
        reward_changed=bool(reward_changed),
        reason=reason,
        source_file=source_file,
        backup_file=backup_file,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Priority 6: Independent Policy Evaluator (pure Python, CLoud-style)
# Scores the trained policy on fuel efficiency, SOC stability, and convergence.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_policy_quality_score(log_path: str, args) -> Dict[str, Any]:
    """Python-based policy quality assessment after each training chunk."""
    from feedback_parser import parse_train_log

    try:
        episodes = parse_train_log(log_path)
    except Exception as e:
        return {"status": "failed", "error": str(e), "score": None}

    if not episodes:
        return {"status": "empty", "score": None}

    # H2 quality (lower h2_100km = better fuel efficiency)
    h2_eps = [ep for ep in episodes if ep.h2_100km is not None]
    h2_vals = sorted([float(ep.h2_100km) for ep in h2_eps])
    h2_mean = sum(h2_vals) / len(h2_vals) if h2_vals else None
    h2_p10 = h2_vals[max(0, len(h2_vals) // 10 - 1)] if h2_vals else None

    # SOC quality (higher in-bounds rate = better)
    inbounds_vals = [float(ep.in_bounds_rate) for ep in episodes if ep.in_bounds_rate is not None]
    inbounds_mean = sum(inbounds_vals) / len(inbounds_vals) if inbounds_vals else None

    # Battery buffer utilization
    soc_swings = []
    for ep in episodes:
        if ep.soc_min is not None and ep.soc_max is not None:
            soc_swings.append(float(ep.soc_max) - float(ep.soc_min))
    mean_swing = sum(soc_swings) / len(soc_swings) if soc_swings else None

    # Convergence: where is the best reward episode?
    reward_eps = [ep for ep in episodes if ep.ep_cumulative_r is not None]
    n = len(reward_eps)
    best_reward = max((float(ep.ep_cumulative_r) for ep in reward_eps), default=None)
    best_ep_idx = next(
        (i for i, ep in enumerate(reward_eps) if float(ep.ep_cumulative_r or float("-inf")) == best_reward),
        None,
    )
    best_ep_ratio = (best_ep_idx / max(1, n - 1)) if (best_ep_idx is not None and n > 1) else 1.0

    # Scores (0-100)
    h2_ref = 2.5  # reference target g/100km
    h2_score = max(0.0, min(100.0, 100.0 * (h2_ref / h2_mean))) if h2_mean else 50.0
    soc_score = float(inbounds_mean) if inbounds_mean is not None else 50.0
    conv_score = max(0.0, 100.0 * (1.0 - best_ep_ratio))  # higher if best found late

    composite = round(0.40 * h2_score + 0.40 * soc_score + 0.20 * conv_score, 1)

    return {
        "status": "ok",
        "score": composite,
        "h2_score": round(h2_score, 1),
        "soc_score": round(soc_score, 1),
        "convergence_score": round(conv_score, 1),
        "h2_mean_g100km": round(h2_mean, 3) if h2_mean is not None else None,
        "h2_best_p10": round(h2_p10, 3) if h2_p10 is not None else None,
        "inbounds_rate_mean": round(inbounds_mean, 1) if inbounds_mean is not None else None,
        "mean_soc_swing": round(mean_swing, 4) if mean_swing is not None else None,
        "battery_underutilized": (mean_swing is not None and mean_swing < 0.05),
        "best_episode_ratio": round(best_ep_ratio, 4),
        "n_episodes": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Priority 5: Structured Iteration Memory helpers
# trajectory_memory records physical metrics; insight_memory distills lessons.
# ─────────────────────────────────────────────────────────────────────────────

def _update_trajectory_memory(
    trajectory_memory: Dict[str, Any],
    iteration: int,
    iter_info: Dict[str, Any],
) -> None:
    """Record current iteration's physical metrics into trajectory_memory."""
    label = f"iter_{iteration:03d}"
    pq = iter_info.get("policy_quality") or {}
    trajectory_memory[label] = {
        "h2_mean": pq.get("h2_mean_g100km"),
        "h2_best_p10": pq.get("h2_best_p10"),
        "inbounds_mean": pq.get("inbounds_rate_mean"),
        "mean_soc_swing": pq.get("mean_soc_swing"),
        "composite_score": pq.get("score"),
        "best_episode_ratio": pq.get("best_episode_ratio"),
        "patched": bool(iter_info.get("patched", False)),
        "final_reason": iter_info.get("final_reason", "unknown"),
    }


def _update_insight_memory(
    insight_memory: List[Dict[str, Any]],
    iteration: int,
    iter_info: Dict[str, Any],
    trajectory_memory: Dict[str, Any],
) -> None:
    """Generate and append a distilled lesson from this iteration's outcome."""
    label = f"iter_{iteration:03d}"
    current = trajectory_memory.get(label, {})
    prev_label = f"iter_{iteration - 1:03d}" if iteration > 1 else None
    prev = trajectory_memory.get(prev_label, {}) if prev_label else {}

    h2_current = current.get("h2_mean")
    h2_prev = prev.get("h2_mean")
    patched = bool(current.get("patched", False))
    reason = str(current.get("final_reason", "unknown"))

    if patched and h2_current is not None and h2_prev is not None:
        delta_pct = (h2_prev - h2_current) / max(abs(h2_prev), 1e-9) * 100.0
        if delta_pct > 1.0:
            lesson = (
                f"Reward patch improved fuel efficiency by {delta_pct:.1f}% "
                f"(h2: {h2_prev:.3f} → {h2_current:.3f} g/100km)"
            )
            outcome_type = "fuel_improvement"
        elif delta_pct < -1.0:
            lesson = (
                f"Reward patch DEGRADED fuel efficiency by {abs(delta_pct):.1f}% "
                f"(h2: {h2_prev:.3f} → {h2_current:.3f} g/100km); investigate cause"
            )
            outcome_type = "fuel_regression"
        else:
            lesson = f"Reward patch applied; fuel efficiency approximately stable ({h2_current:.3f} g/100km)"
            outcome_type = "neutral"
    elif not patched:
        lesson = f"No patch applied this iteration ({reason}); current reward carried forward"
        outcome_type = "no_change"
    else:
        lesson = "Reward patched; fuel efficiency data not available for this iteration"
        outcome_type = "patched_no_data"

    insight_memory.append({
        "iteration": iteration,
        "label": label,
        "outcome_type": outcome_type,
        "lesson": lesson,
        "patched": patched,
        "h2_mean": h2_current,
        "inbounds_mean": current.get("inbounds_mean"),
        "mean_soc_swing": current.get("mean_soc_swing"),
        "composite_score": current.get("composite_score"),
    })


def _load_existing_chunk_result(run_root: Path, args, source_iteration: int, chunk_result_cls):
    source_iter_dir = run_root / f"iter_{int(source_iteration):03d}"
    log_path = source_iter_dir / "train_log.log"
    episode_data_dir = source_iter_dir / "episode_data"
    if not log_path.exists():
        raise RuntimeError(f"manual pre-refine source log missing: {log_path}")
    if not episode_data_dir.exists():
        raise RuntimeError(
            "manual pre-refine source episode_data missing: "
            f"{episode_data_dir}"
        )

    start_episode = int(getattr(args, "start_episode", 0))
    end_episode = start_episode + max(0, int(args.chunk_episodes) - 1)
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"chunk finished:\s*episodes\s*(\d+)\.\.(\d+)", text)
        if m:
            start_episode = int(m.group(1))
            end_episode = int(m.group(2))
        else:
            ep_matches = re.findall(r"^epi\s+(\d+):", text, flags=re.MULTILINE)
            if ep_matches:
                start_episode = int(ep_matches[0])
                end_episode = int(ep_matches[-1])
    except Exception:
        # Fallback to default episode window when parsing is unavailable.
        pass

    model_dir = (
        run_root
        / "models"
        / f"iter_{int(source_iteration):03d}"
        / args.scenario_name
        / "net_params"
    )
    return chunk_result_cls(
        iteration=int(source_iteration),
        start_episode=int(start_episode),
        end_episode=int(end_episode),
        log_path=str(log_path),
        episode_data_dir=str(episode_data_dir),
        model_dir=str(model_dir),
        chunk_dir=str(source_iter_dir),
        best_episode=None,
        best_reward=None,
    )


def main():
    args = get_ldri_args()

    if int(getattr(args, "resume_additional_iterations", 0)) < 0:
        raise RuntimeError("--resume_additional_iterations must be >= 0")
    if int(getattr(args, "manual_pre_refine_source_iter", 0)) < 0:
        raise RuntimeError("--manual_pre_refine_source_iter must be >= 0")

    run_root, resume_mode = _resolve_run_root(args)
    summary_path = run_root / "run_summary.json"
    run_config_path = run_root / "run_config.json"

    if resume_mode and bool(getattr(args, "resume_use_saved_config", True)):
        if run_config_path.exists():
            saved_run_config = _read_json(run_config_path)
            overridden = _apply_resume_run_config(args, saved_run_config)
            if overridden:
                print("[resume] restored configuration from run_config.json")
                for key, old_val, new_val in overridden:
                    print(f"  - {key}: {old_val!r} -> {new_val!r}")
        else:
            print(
                "[resume] warning: run_config.json not found; "
                "continuing with current CLI arguments."
            )

    args.reward_file = _resolve_input_path(args.reward_file)
    args.prompt_template = _resolve_input_path(args.prompt_template)
    args.bootstrap_prompt_template = _resolve_input_path(args.bootstrap_prompt_template)
    ensure_lr_reference_attrs(args)
    _configure_training_paths(args, run_root)

    if resume_mode:
        if args.bootstrap_only:
            raise RuntimeError("--bootstrap_only cannot be used with --resume_run_root")
        if not summary_path.exists():
            raise RuntimeError(
                f"resume requires existing run_summary.json: {summary_path}"
            )

        summary = _read_json(summary_path)
        if not isinstance(summary.get("iterations"), list):
            raise RuntimeError("invalid run_summary.json: 'iterations' must be a list")
        summary["run_root"] = str(run_root)

        existing_iterations = _extract_iteration_numbers(summary)
        last_iteration = max(existing_iterations) if existing_iterations else 0
        requested_total = int(args.ldri_iterations)
        additional = int(getattr(args, "resume_additional_iterations", 0))
        if additional > 0:
            iteration_end = last_iteration + additional
        else:
            iteration_end = requested_total
        iteration_start = last_iteration + 1

        if iteration_end < iteration_start:
            print("No additional iterations to run in resume mode.")
            print(
                f"Existing last iteration: {last_iteration}, "
                f"requested total iterations: {iteration_end}"
            )
            print(f"Artifacts: {run_root}")
            return

        summary.setdefault("resume_events", [])
        summary["resume_events"].append(
            {
                "resumed_at": dt.datetime.now().isoformat(),
                "previous_last_iteration": last_iteration,
                "start_iteration": iteration_start,
                "end_iteration": iteration_end,
                "requested_ldri_iterations": requested_total,
                "resume_additional_iterations": additional,
            }
        )
        _write_json(summary_path, summary)

        if run_config_path.exists():
            run_config = _read_json(run_config_path)
            run_config.pop("tpe_dispreferred_in_bounds_floor", None)
            prev_total = int(run_config.get("ldri_iterations", 0) or 0)
            run_config["ldri_iterations"] = max(prev_total, int(iteration_end))
            run_config["last_resumed_at"] = dt.datetime.now().isoformat()
            _write_json(run_config_path, run_config)
    else:
        if not args.bootstrap_from_scratch:
            raise RuntimeError(
                "--bootstrap_from_scratch must be True. "
                "LDRI workflow now requires scratch bootstrap before training."
            )
        if args.skip_llm:
            raise RuntimeError("bootstrap requires LLM calls; remove --skip_llm")
        if args.skip_patch:
            raise RuntimeError("bootstrap requires patch apply; remove --skip_patch")
        if args.bootstrap_only and not args.bootstrap_from_scratch:
            raise RuntimeError("--bootstrap_only requires --bootstrap_from_scratch True")

        run_config = {
            "DRL": args.DRL,
            "scenario_name": args.scenario_name,
            "MODE": args.MODE,
            "soc0": args.soc0,
            "soc_target": args.soc_target,
            "w_soc": args.w_soc,
            "w_h2": args.w_h2,
            "w_fc": args.w_fc,
            "w_batt": args.w_batt,
            "eq_h2_batt_coef": args.eq_h2_batt_coef,
            "lr_critic": args.lr_critic,
            "lr_critic_min": args.lr_critic_min,
            "lr_actor": args.lr_actor,
            "lr_actor_min": args.lr_actor_min,
            "lr_alpha": args.lr_alpha,
            "lr_schedule_episodes": args.lr_schedule_episodes,
            "lr_restart_cycles": args.lr_restart_cycles,
            "lr_warmup_ratio": args.lr_warmup_ratio,
            "auto_lr_on_reward_patch": args.auto_lr_on_reward_patch,
            "lr_auto_scale_floor": args.lr_auto_scale_floor,
            "lr_auto_scale_ceiling": args.lr_auto_scale_ceiling,
            "lr_auto_warmup_min": args.lr_auto_warmup_min,
            "lr_auto_warmup_max": args.lr_auto_warmup_max,
            "lr_auto_warmup_gain": args.lr_auto_warmup_gain,
            "ldri_iterations": args.ldri_iterations,
            "chunk_episodes": args.chunk_episodes,
            "feedback_process_episodes": args.feedback_process_episodes,
            "reset_agent_on_patch": args.reset_agent_on_patch,
            "reset_episode_counter_on_patch": args.reset_episode_counter_on_patch,
            "reward_module": args.reward_module,
            "reward_file": args.reward_file,
            "prompt_template": args.prompt_template,
            "bootstrap_from_scratch": args.bootstrap_from_scratch,
            "bootstrap_only": args.bootstrap_only,
            "bootstrap_prompt_template": args.bootstrap_prompt_template,
            "bootstrap_retries": args.bootstrap_retries,
            "bootstrap_task_description": args.bootstrap_task_description,
            "bootstrap_max_attrs": args.bootstrap_max_attrs,
            "bootstrap_max_methods": args.bootstrap_max_methods,
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "multi_agent_review": args.multi_agent_review,
            "parse_retries": args.parse_retries,
            "tpe_retries": args.tpe_retries,
            "tpe_soc_in_bounds_floor": args.tpe_soc_in_bounds_floor,
            "soc_low": args.soc_low,
            "soc_high": args.soc_high,
            "aligned_min_tac": args.aligned_min_tac,
            "aligned_dominance_threshold": args.aligned_dominance_threshold,
            "aligned_soc_dominance_threshold": args.aligned_soc_dominance_threshold,
            "aligned_min_h2_fraction": args.aligned_min_h2_fraction,
            "aligned_h2_perturbation_ratio": args.aligned_h2_perturbation_ratio,
            "created_at": dt.datetime.now().isoformat(),
        }
        _write_json(run_config_path, run_config)

        summary = {
            "run_root": str(run_root),
            "bootstrap": None,
            "iterations": [],
        }
        _write_json(summary_path, summary)
        iteration_start = 1
        iteration_end = int(args.ldri_iterations)

    from env_ldri import make_env_ldri
    from runner_ldri import ChunkResult, RunnerLDRI
    llm_client = _init_llm_client(args)

    env, args = make_env_ldri(args, force_reload_agent=True)
    runner = RunnerLDRI(args, env, run_root=run_root)

    # Priority 5: Structured Iteration Memory — initialized once, updated each iteration
    trajectory_memory: Dict[str, Any] = {}
    insight_memory: List[Dict[str, Any]] = []

    print(f"LDRI run root: {run_root}")
    print(f"Training algorithm: {args.DRL}")
    print(f"Reward module/file: {args.reward_module} / {args.reward_file}")

    manual_pre_refine_source_iter = int(getattr(args, "manual_pre_refine_source_iter", 0))
    if manual_pre_refine_source_iter > 0:
        if not resume_mode:
            raise RuntimeError(
                "--manual_pre_refine_source_iter is supported only with --resume_run_root"
            )
        if int(iteration_end) != int(iteration_start):
            raise RuntimeError(
                "manual pre-refine currently supports exactly one target iteration. "
                "Use --resume_additional_iterations 1."
            )
        if manual_pre_refine_source_iter >= int(iteration_start):
            raise RuntimeError(
                "manual pre-refine source must be less than target iteration. "
                f"source={manual_pre_refine_source_iter}, target={int(iteration_start)}"
            )
        print(
            "Manual pre-refine mode enabled: "
            f"source iter_{manual_pre_refine_source_iter:03d} -> "
            f"target iter_{int(iteration_start):03d}"
        )

    if resume_mode:
        print(
            f"Resume mode: running iterations {iteration_start}..{iteration_end} "
            f"(existing run preserved)"
        )
    else:
        print("\n===== LDRI Bootstrap: Scratch Reward Generation =====")
        bootstrap_result = _run_bootstrap_from_scratch(
            args=args,
            client=llm_client,
            env=env,
            run_root=run_root,
        )
        summary["bootstrap"] = bootstrap_result
        _write_json(summary_path, summary)

        if not bootstrap_result.get("patched", False):
            raise RuntimeError(
                "bootstrap_from_scratch failed to produce a patchable reward method. "
                "Check artifacts in run_root/bootstrap."
            )

        env, args = make_env_ldri(args, force_reload_agent=True)
        runner.set_env(env)
        if args.flush_buffer_on_patch:
            runner.reset_buffer()
            summary["bootstrap_buffer_flushed"] = True
        else:
            summary["bootstrap_buffer_flushed"] = False
        _write_json(summary_path, summary)

        if args.bootstrap_only:
            summary["workflow_status"] = "bootstrap_only_completed"
            _write_json(summary_path, summary)
            print("\nBootstrap-only workflow finished.")
            print(f"Bootstrap artifacts: {run_root / 'bootstrap'}")
            print(f"Run summary: {summary_path}")
            return

    for iteration in range(int(iteration_start), int(iteration_end) + 1):
        print(f"\n===== LDRI Iteration {iteration}/{iteration_end} =====")
        iter_dir = run_root / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        observer = LDRIObserver(iter_dir=iter_dir, iteration=iteration)

        if manual_pre_refine_source_iter > 0:
            source_chunk = _load_existing_chunk_result(
                run_root=run_root,
                args=args,
                source_iteration=manual_pre_refine_source_iter,
                chunk_result_cls=ChunkResult,
            )
            observer.status(
                "MANUAL_PRE_REFINE_SOURCE",
                (
                    f"using iter_{manual_pre_refine_source_iter:03d} "
                    f"artifacts ({source_chunk.start_episode}..{source_chunk.end_episode})"
                ),
            )

            feedback_bundle = build_feedback_bundle(
                log_path=source_chunk.log_path,
                episode_data_dir=source_chunk.episode_data_dir,
                process_episodes=args.feedback_process_episodes,
                soc_bounds=(args.soc_low, args.soc_high),
                preference_in_bounds_floor=args.tpe_soc_in_bounds_floor,
            )
            _write_text(iter_dir / "process_feedback.txt", feedback_bundle["PROCESS_FEEDBACK"])
            _write_text(iter_dir / "trajectory_feedback.txt", feedback_bundle["TRAJECTORY_FEEDBACK"])
            _write_text(iter_dir / "preference_feedback.txt", feedback_bundle["PREFERENCE_FEEDBACK"])
            observer.status(
                "FEEDBACK_BUILT",
                (
                    "process/trajectory/preference feedback saved "
                    f"from iter_{manual_pre_refine_source_iter:03d}"
                ),
            )

            current_method = extract_method_source(
                args.reward_file,
                class_name="EMS",
                method_name="get_reward",
            )
            reward_history = _collect_trained_reward_history(
                run_root,
                upto_iteration=iteration,
            )
            _register_reward_method(
                reward_history,
                current_method,
                origin=f"current_iter_{iteration:03d}_pre_refine",
                source_file=Path(args.reward_file),
            )
            prompt = _build_prompt(
                args,
                feedback_bundle,
                current_method,
                iteration,
                reward_history=reward_history,
            )
            _write_text(iter_dir / "prompt.md", prompt)
            observer.status("PROMPT_SENT", "prompt.md rendered for manual pre-refine")

            iter_info: Dict[str, Any] = {
                "iteration": iteration,
                "chunk": None,
                "candidate_count": 0,
                "best_tpe_delta": None,
                "patched": False,
            }

            final_reason = "skip_llm_pretrain"
            final_reward_changed = False
            final_source_file = args.reward_file
            final_backup_file = None

            if args.skip_llm:
                observer.status("SKIPPED", "skip_llm=True")
            else:
                # Load prev_method (shared by both paths)
                _prev_method_pretrain: Optional[str] = None
                if iteration > 1:
                    _prev_iter_path = iter_dir.parent / f"iter_{iteration - 1:03d}"
                    _prev_cand = _load_iteration_candidate_file(_prev_iter_path)
                    if _prev_cand is not None:
                        try:
                            _prev_method_pretrain = _prev_cand.read_text(encoding="utf-8")
                        except Exception:
                            pass

                if getattr(args, "multi_agent_review", False):
                    _iter_memory_pretrain = {
                        "trajectory": dict(trajectory_memory),
                        "insights": list(insight_memory[-3:]),
                    }
                    llm_result = run_multi_agent_review(
                        args=args,
                        client=llm_client,
                        feedback_bundle=feedback_bundle,
                        current_method=current_method,
                        iter_dir=iter_dir,
                        chunk_result=source_chunk,
                        observer=observer,
                        reward_history=reward_history,
                        prev_method=_prev_method_pretrain,
                        iteration_memory=_iter_memory_pretrain,
                        policy_quality=None,
                    )
                    # Merge reviewer blackboard posts into shared insight_memory (① blackboard)
                    for bp in llm_result.get("reviewer_blackboard_posts", []):
                        bp["iteration"] = iteration
                        insight_memory.append(bp)
                else:
                    llm_result = _run_llm_with_tpe_retry(
                        args=args,
                        client=llm_client,
                        prompt=prompt,
                        iter_dir=iter_dir,
                        chunk_result=source_chunk,
                        observer=observer,
                        current_method=current_method,
                        reward_history=reward_history,
                        feedback_bundle=feedback_bundle,
                        iteration=iteration,
                        prev_method=_prev_method_pretrain,
                    )

                iter_info["candidate_count"] = llm_result["parse_ok_count"]
                iter_info["best_tpe_delta"] = llm_result["best_tpe_delta"]
                iter_info["best_candidate_attempt"] = llm_result["best_candidate_attempt"]

                if llm_result["best_candidate_method"] is not None:
                    diff_text = method_diff_preview(
                        current_method,
                        llm_result["best_candidate_method"],
                    )
                    _write_text(iter_dir / "best_candidate_vs_current.diff", diff_text)

                passed_candidate = llm_result["passed_candidate_method"]
                if passed_candidate is None:
                    if llm_result["parse_ok_count"] == 0:
                        reason = "parse_fail"
                    elif llm_result["tpe_eval_count"] == 0 and llm_result["historical_duplicate_count"] > 0:
                        reason = "historical_duplicate"
                    else:
                        reason = "tpe_fail"
                    observer.status("SKIPPED", f"{reason}: keeping current reward")
                    final_reason = f"{reason}_pretrain"
                elif args.skip_patch:
                    observer.status("SKIPPED", "skip_patch=True")
                    final_reason = "skip_patch_pretrain"
                else:
                    diff_text = method_diff_preview(current_method, passed_candidate)
                    _write_text(iter_dir / "candidate_method.diff", diff_text)

                    patch_result = patch_method_in_file(
                        file_path=args.reward_file,
                        method_source=passed_candidate,
                        class_name="EMS",
                        method_name="get_reward",
                        required_info_keys=REQUIRED_INFO_KEYS,
                        backup=True,
                    )
                    subprocess.run(["python3", "-m", "py_compile", args.reward_file], check=True)

                    iter_info["patched"] = True
                    iter_info["patch_target"] = patch_result.target_file
                    iter_info["backup_file"] = patch_result.backup_file
                    observer.status("PATCHED", f"file={patch_result.target_file}")

                    _maybe_auto_tune_lr_after_patch(
                        args=args,
                        observer=observer,
                        iter_info=iter_info,
                        current_method=current_method,
                        candidate_method=passed_candidate,
                        chunk_result=source_chunk,
                        lr_proposal=llm_result.get("trainability_lr_proposal"),
                    )

                    env, args = make_env_ldri(args, force_reload_agent=True)
                    runner.set_env(env)

                    if args.reset_agent_on_patch:
                        runner.reset_learning_state(
                            reset_episode_counter=bool(args.reset_episode_counter_on_patch)
                        )
                        iter_info["agent_reset"] = True
                        iter_info["buffer_flushed"] = True
                        iter_info["episode_counter_reset"] = bool(
                            args.reset_episode_counter_on_patch
                        )
                        observer.status(
                            "AGENT_RESET",
                            (
                                "model parameters re-initialized after patch; "
                                f"episode_counter_reset={bool(args.reset_episode_counter_on_patch)}"
                            ),
                        )
                    elif args.flush_buffer_on_patch:
                        runner.reset_buffer()
                        iter_info["agent_reset"] = False
                        iter_info["buffer_flushed"] = True
                        iter_info["episode_counter_reset"] = False
                    else:
                        iter_info["agent_reset"] = False
                        iter_info["buffer_flushed"] = False
                        iter_info["episode_counter_reset"] = False

                    final_reason = "passed_and_patched_pretrain"
                    final_reward_changed = True
                    final_source_file = patch_result.target_file
                    final_backup_file = patch_result.backup_file

            chunk = runner.train_chunk(iteration=iteration, chunk_episodes=args.chunk_episodes)
            observer.status("TRAINED", f"episodes {chunk.start_episode}..{chunk.end_episode}")
            # Priority 6: policy quality for manual pre-refine path
            pq_pretrain = _compute_policy_quality_score(chunk.log_path, args)
            _write_json(iter_dir / "policy_quality.json", pq_pretrain)
            iter_info["policy_quality"] = pq_pretrain
            iter_info["chunk"] = {
                "start_episode": chunk.start_episode,
                "end_episode": chunk.end_episode,
                "episodes": (chunk.end_episode - chunk.start_episode + 1),
                "log_path": chunk.log_path,
                "episode_data_dir": chunk.episode_data_dir,
                "model_dir": chunk.model_dir,
            }

            _finalize_iteration(
                observer=observer,
                iter_info=iter_info,
                reason=final_reason,
                reward_changed=final_reward_changed,
                source_file=final_source_file,
                backup_file=final_backup_file,
            )
            _update_trajectory_memory(trajectory_memory, iteration, iter_info)
            _update_insight_memory(insight_memory, iteration, iter_info, trajectory_memory)
            summary["iterations"].append(iter_info)
            _write_json(summary_path, summary)
            continue

        # 1) current reward로 학습
        chunk = runner.train_chunk(iteration=iteration, chunk_episodes=args.chunk_episodes)
        observer.status("TRAINED", f"episodes {chunk.start_episode}..{chunk.end_episode}")

        # Priority 6: Independent Policy Evaluator — score the trained policy
        policy_quality = _compute_policy_quality_score(chunk.log_path, args)
        _write_json(iter_dir / "policy_quality.json", policy_quality)
        observer.status(
            "POLICY_QUALITY",
            f"score={policy_quality.get('score')} "
            f"h2={policy_quality.get('h2_mean_g100km')} "
            f"swing={policy_quality.get('mean_soc_swing')}",
        )

        # 2) 3가지 종류의 feedback 생성
        feedback_bundle = build_feedback_bundle(
            log_path=chunk.log_path,
            episode_data_dir=chunk.episode_data_dir,
            process_episodes=args.feedback_process_episodes,
            soc_bounds=(args.soc_low, args.soc_high),
            preference_in_bounds_floor=args.tpe_soc_in_bounds_floor,
        )
        _write_text(iter_dir / "process_feedback.txt", feedback_bundle["PROCESS_FEEDBACK"])
        _write_text(iter_dir / "trajectory_feedback.txt", feedback_bundle["TRAJECTORY_FEEDBACK"])
        _write_text(iter_dir / "preference_feedback.txt", feedback_bundle["PREFERENCE_FEEDBACK"])
        observer.status("FEEDBACK_BUILT", "process/trajectory/preference feedback saved")

        # 3) Reward refine시 LLM에 직접적으로 입력될 prompt 생성
        current_method = extract_method_source(args.reward_file, class_name="EMS", method_name="get_reward")
        reward_history = _collect_trained_reward_history(
            run_root,
            upto_iteration=iteration,
        )
        _register_reward_method(
            reward_history,
            current_method,
            origin=f"current_iter_{iteration:03d}_pre_refine",
            source_file=Path(args.reward_file),
        )
        prompt = _build_prompt(
            args,
            feedback_bundle,
            current_method,
            iteration,
            reward_history=reward_history,
        )
        _write_text(iter_dir / "prompt.md", prompt)
        observer.status("PROMPT_SENT", "prompt.md rendered for this iteration")

        iter_info: Dict[str, Any] = {
            "iteration": iteration,
            "chunk": {
                "start_episode": chunk.start_episode,
                "end_episode": chunk.end_episode,
                "episodes": (chunk.end_episode - chunk.start_episode + 1),
                "log_path": chunk.log_path,
                "episode_data_dir": chunk.episode_data_dir,
                "model_dir": chunk.model_dir,
            },
            "candidate_count": 0,
            "best_tpe_delta": None,
            "patched": False,
            "policy_quality": policy_quality,
        }

        # 4) LLM이 reward를 refine한 후보군을 제시함
        if args.skip_llm:
            observer.status("SKIPPED", "skip_llm=True")
            _finalize_iteration(
                observer=observer,
                iter_info=iter_info,
                reason="skip_llm",
                reward_changed=False,
                source_file=args.reward_file,
                backup_file=None,
            )
            _update_trajectory_memory(trajectory_memory, iteration, iter_info)
            _update_insight_memory(insight_memory, iteration, iter_info, trajectory_memory)
            summary["iterations"].append(iter_info)
            _write_json(summary_path, summary)
            continue

        # Load prev_method (shared by both multi-agent and single-agent paths)
        _prev_method_main: Optional[str] = None
        if iteration > 1:
            _prev_iter_path = iter_dir.parent / f"iter_{iteration - 1:03d}"
            _prev_cand = _load_iteration_candidate_file(_prev_iter_path)
            if _prev_cand is not None:
                try:
                    _prev_method_main = _prev_cand.read_text(encoding="utf-8")
                except Exception:
                    pass

        if getattr(args, "multi_agent_review", False):
            _iter_memory = {
                "trajectory": dict(trajectory_memory),
                "insights": list(insight_memory[-3:]),
            }
            llm_result = run_multi_agent_review(
                args=args,
                client=llm_client,
                feedback_bundle=feedback_bundle,
                current_method=current_method,
                iter_dir=iter_dir,
                chunk_result=chunk,
                observer=observer,
                reward_history=reward_history,
                prev_method=_prev_method_main,
                iteration_memory=_iter_memory,
                policy_quality=policy_quality,
            )
            # Merge reviewer blackboard posts into shared insight_memory (① blackboard)
            for bp in llm_result.get("reviewer_blackboard_posts", []):
                bp["iteration"] = iteration
                insight_memory.append(bp)
        else:
            llm_result = _run_llm_with_tpe_retry(
                args=args,
                client=llm_client,
                prompt=prompt,
                iter_dir=iter_dir,
                chunk_result=chunk,
                observer=observer,
                current_method=current_method,
                reward_history=reward_history,
                feedback_bundle=feedback_bundle,
                iteration=iteration,
                prev_method=_prev_method_main,
            )

        iter_info["candidate_count"] = llm_result["parse_ok_count"]
        iter_info["best_tpe_delta"] = llm_result["best_tpe_delta"]
        iter_info["best_candidate_attempt"] = llm_result["best_candidate_attempt"]

        # Always generate diff if at least one parseable candidate exists.
        if llm_result["best_candidate_method"] is not None:
            diff_text = method_diff_preview(current_method, llm_result["best_candidate_method"])
            _write_text(iter_dir / "best_candidate_vs_current.diff", diff_text)

        passed_candidate = llm_result["passed_candidate_method"]

        if passed_candidate is None:
            if llm_result["parse_ok_count"] == 0:
                reason = "parse_fail"
            elif llm_result["tpe_eval_count"] == 0 and llm_result["historical_duplicate_count"] > 0:
                reason = "historical_duplicate"
            else:
                reason = "tpe_fail"
            observer.status("SKIPPED", f"{reason}: keeping current reward")
            _finalize_iteration(
                observer=observer,
                iter_info=iter_info,
                reason=reason,
                reward_changed=False,
                source_file=args.reward_file,
                backup_file=None,
            )
            _update_trajectory_memory(trajectory_memory, iteration, iter_info)
            _update_insight_memory(insight_memory, iteration, iter_info, trajectory_memory)
            summary["iterations"].append(iter_info)
            _write_json(summary_path, summary)
            continue

        # 5) TPE를 통과하면 해당 reward로 업데이트하고 다음 iteration 진행 (only EMS.get_reward)
        if args.skip_patch:
            observer.status("SKIPPED", "skip_patch=True")
            _finalize_iteration(
                observer=observer,
                iter_info=iter_info,
                reason="skip_patch",
                reward_changed=False,
                source_file=args.reward_file,
                backup_file=None,
            )
            _update_trajectory_memory(trajectory_memory, iteration, iter_info)
            _update_insight_memory(insight_memory, iteration, iter_info, trajectory_memory)
            summary["iterations"].append(iter_info)
            _write_json(summary_path, summary)
            continue

        diff_text = method_diff_preview(current_method, passed_candidate)
        _write_text(iter_dir / "candidate_method.diff", diff_text)

        patch_result = patch_method_in_file(
            file_path=args.reward_file,
            method_source=passed_candidate,
            class_name="EMS",
            method_name="get_reward",
            required_info_keys=REQUIRED_INFO_KEYS,
            backup=True,
        )
        subprocess.run(["python3", "-m", "py_compile", args.reward_file], check=True)

        iter_info["patched"] = True
        iter_info["patch_target"] = patch_result.target_file
        iter_info["backup_file"] = patch_result.backup_file

        observer.status("PATCHED", f"file={patch_result.target_file}")

        _maybe_auto_tune_lr_after_patch(
            args=args,
            observer=observer,
            iter_info=iter_info,
            current_method=current_method,
            candidate_method=passed_candidate,
            chunk_result=chunk,
            lr_proposal=llm_result.get("trainability_lr_proposal"),
        )

        # 6) env module 다시 로드해서 새로운 reward로 학습
        env, args = make_env_ldri(args, force_reload_agent=True)
        runner.set_env(env)

        if args.reset_agent_on_patch:
            runner.reset_learning_state(
                reset_episode_counter=bool(args.reset_episode_counter_on_patch)
            )
            iter_info["agent_reset"] = True
            iter_info["buffer_flushed"] = True
            iter_info["episode_counter_reset"] = bool(args.reset_episode_counter_on_patch)
            observer.status(
                "AGENT_RESET",
                (
                    "model parameters re-initialized after patch; "
                    f"episode_counter_reset={bool(args.reset_episode_counter_on_patch)}"
                ),
            )
        elif args.flush_buffer_on_patch:
            runner.reset_buffer()
            iter_info["agent_reset"] = False
            iter_info["buffer_flushed"] = True
            iter_info["episode_counter_reset"] = False
        else:
            iter_info["agent_reset"] = False
            iter_info["buffer_flushed"] = False
            iter_info["episode_counter_reset"] = False

        _finalize_iteration(
            observer=observer,
            iter_info=iter_info,
            reason="passed_and_patched",
            reward_changed=True,
            source_file=patch_result.target_file,
            backup_file=patch_result.backup_file,
        )
        _update_trajectory_memory(trajectory_memory, iteration, iter_info)
        _update_insight_memory(insight_memory, iteration, iter_info, trajectory_memory)

        summary["iterations"].append(iter_info)
        _write_json(summary_path, summary)

    print("\nLDRI workflow finished.")
    print(f"Artifacts: {run_root}")


if __name__ == "__main__":
    main()
