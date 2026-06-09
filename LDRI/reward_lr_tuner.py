from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as scio

from feedback_parser import parse_train_log
from ldri_tpe_feasible_pool import select_feasible_pool


class _StubEMS:
    pass


def _clip(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _safe_ratio(new_value: Optional[float], old_value: Optional[float], default: float = 1.0) -> float:
    try:
        new_v = float(new_value)
        old_v = float(old_value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(new_v) or not math.isfinite(old_v) or abs(old_v) < 1e-12:
        return float(default)
    return max(1e-9, float(new_v / old_v))


def ensure_lr_reference_attrs(args) -> None:
    if not hasattr(args, "lr_warmup_ratio"):
        args.lr_warmup_ratio = 0.10

    base_pairs = [
        ("lr_critic_base", "lr_critic"),
        ("lr_critic_min_base", "lr_critic_min"),
        ("lr_actor_base", "lr_actor"),
        ("lr_actor_min_base", "lr_actor_min"),
        ("lr_warmup_ratio_base", "lr_warmup_ratio"),
    ]
    for base_name, current_name in base_pairs:
        if not hasattr(args, base_name):
            setattr(args, base_name, float(getattr(args, current_name)))

    if not hasattr(args, "lr_auto_factor"):
        base = max(1e-12, float(getattr(args, "lr_critic_base")))
        args.lr_auto_factor = float(getattr(args, "lr_critic")) / base


def _arr(mat, *keys, default_len=None):
    for key in keys:
        if key in mat:
            arr = np.asarray(mat[key], dtype=np.float64).squeeze()
            if arr.ndim == 0:
                return np.array([float(arr)], dtype=np.float64)
            return arr.reshape(-1)
    if default_len is None:
        return np.array([], dtype=np.float64)
    return np.zeros(int(default_len), dtype=np.float64)


def _build_candidate_fn(method_source: str):
    namespace = {"np": np}
    code = textwrap.dedent(method_source)
    exec(code, namespace, namespace)
    fn = namespace.get("get_reward")
    if fn is None:
        raise RuntimeError("method source does not define callable get_reward(self)")
    return fn


def _reward_values_on_episode(fn, mat_path: Path, args) -> np.ndarray:
    mat = scio.loadmat(str(mat_path))
    soc = _arr(mat, "SOC")
    n = len(soc)
    if n == 0:
        raise RuntimeError(f"SOC array missing/empty in {mat_path}")

    h2_fcs = _arr(mat, "h2_fcs", default_len=n)
    p_batt_kw = _arr(mat, "P_batt_req", "pack_power_out", default_len=n)
    p_fc_kw = _arr(mat, "P_fc", default_len=n)
    dsoh_fcs = _arr(mat, "FCS_De", "fcs_soh_cost", default_len=n)
    dsoh_batt = _arr(mat, "dsoh", "batt_soh_cost", default_len=n)

    stub = _StubEMS()
    stub.w_soc = float(args.w_soc)
    stub.w_h2 = float(getattr(args, "w_h2", 10.0))
    stub.w_fc = float(getattr(args, "w_fc", 4.275e6))
    stub.w_batt = float(getattr(args, "w_batt", 1.116e6))
    stub.SOC_target = float(args.soc_target)
    stub.soc_bounds = (float(args.soc_low), float(args.soc_high))
    stub.reward_scale = float(getattr(args, "reward_scale", 1.0))
    stub.soc_weight_multiplier = 1.0
    stub.eq_h2_batt_coef = float(getattr(args, "eq_h2_batt_coef", 0.0164))
    stub.time_step = float(getattr(args, "time_step", 1.0))
    stub.P_batt_max = max(1.0, float(np.nanmax(np.abs(p_batt_kw))) * 1000.0)
    stub.P_FCS_max = max(1.0, float(np.nanmax(np.abs(p_fc_kw))))
    stub.info = {}

    rewards: List[float] = []
    for i in range(n):
        stub.SOC = float(soc[i])
        stub.h2_fcs = float(h2_fcs[i])
        stub.P_batt = float(p_batt_kw[i]) * 1000.0
        stub.P_FCS = float(p_fc_kw[i])
        stub.P_fc = stub.P_FCS
        stub.dSOH_FCS = float(dsoh_fcs[i])
        stub.dSOH_batt = float(dsoh_batt[i])
        stub.h2_batt = stub.P_batt / 1000.0 * stub.eq_h2_batt_coef
        stub.h2_equal = stub.h2_fcs + stub.h2_batt
        stub.info = {}

        value = fn(stub)
        if isinstance(value, tuple):
            value = value[0]
        rewards.append(float(value))
    return np.asarray(rewards, dtype=np.float64)


def _pool_args(args) -> Tuple[float, float, int, int, int]:
    soc_target = float(getattr(args, "soc_target", 0.6))
    min_in_bounds_rate = float(getattr(args, "tpe_soc_in_bounds_floor", 100.0))
    pool_size = max(4, int(round(float(getattr(args, "tpe_fp_pool_size", 6)))))
    topk = max(1, int(round(float(getattr(args, "tpe_fp_topk", 2)))))
    bottomk = max(1, int(round(float(getattr(args, "tpe_fp_bottomk", 2)))))
    return soc_target, min_in_bounds_rate, pool_size, topk, bottomk


def _select_profile_episodes(log_path: str | Path, args) -> Tuple[List[int], Optional[int], Optional[int], str]:
    episodes = parse_train_log(log_path)
    if not episodes:
        return [], None, None, "empty train log"

    soc_target, min_in_bounds_rate, pool_size, topk, bottomk = _pool_args(args)
    selection, status = select_feasible_pool(
        episodes,
        soc_target=soc_target,
        min_in_bounds_rate=min_in_bounds_rate,
        pool_size=pool_size,
        topk=topk,
        bottomk=bottomk,
    )

    ordered_ids: List[int] = []

    def _add_episode_id(episode_id: int) -> None:
        ep_id = int(episode_id)
        if ep_id not in ordered_ids:
            ordered_ids.append(ep_id)

    preferred_episode = None
    dispreferred_episode = None
    if selection is not None:
        preferred_episode = int(selection.best.episode)
        dispreferred_episode = int(selection.degraded.episode)
        for ep in [selection.best, selection.degraded] + list(selection.top_pool) + list(selection.bottom_pool):
            _add_episode_id(int(ep.episode))

    recent_count = max(2, int(getattr(args, "feedback_process_episodes", 10) or 10))
    for ep in episodes[-min(len(episodes), recent_count) :]:
        _add_episode_id(int(ep.episode))

    if selection is None:
        status = (
            "feasible-pool selection unavailable; "
            f"fallback to last {min(len(episodes), recent_count)} episodes"
        )
    else:
        status = f"{status}; plus last {min(len(episodes), recent_count)} episodes"
    return ordered_ids, preferred_episode, dispreferred_episode, status


def _training_trend_summary(log_path: str | Path, args) -> Dict[str, float]:
    episodes = [
        ep
        for ep in parse_train_log(log_path)
        if ep.ep_cumulative_r is not None
    ]
    if len(episodes) < 5:
        return {
            "best_episode_ratio": 1.0,
            "tail_gap_ratio": 0.0,
            "trend_factor": 1.0,
        }

    start_ep = int(episodes[0].episode)
    end_ep = int(episodes[-1].episode)
    horizon = max(1, end_ep - start_ep + 1)
    best_ep = max(episodes, key=lambda ep: float(ep.ep_cumulative_r))
    best_reward = float(best_ep.ep_cumulative_r)
    best_episode_ratio = float(best_ep.episode - start_ep + 1) / float(horizon)

    tail_window = min(
        len(episodes),
        max(5, int(getattr(args, "feedback_process_episodes", 10) or 10)),
    )
    tail_rewards = [float(ep.ep_cumulative_r) for ep in episodes[-tail_window:]]
    tail_avg = float(np.mean(tail_rewards))
    tail_gap_ratio = max(0.0, best_reward - tail_avg) / max(1.0, abs(best_reward))

    early_best_threshold = 0.35
    tail_gap_threshold = 0.06
    early_penalty = _clip(
        (early_best_threshold - best_episode_ratio) / early_best_threshold,
        0.0,
        1.0,
    )
    drift_penalty = 0.0
    if best_episode_ratio < 0.5:
        drift_penalty = _clip(tail_gap_ratio / tail_gap_threshold, 0.0, 1.0)

    trend_factor = max(0.70, (1.0 - 0.20 * early_penalty) * (1.0 - 0.25 * drift_penalty))
    return {
        "best_episode_ratio": float(best_episode_ratio),
        "tail_gap_ratio": float(tail_gap_ratio),
        "trend_factor": float(trend_factor),
    }


def _summarize_values(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    abs_arr = np.abs(arr)
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "abs_mean": float(np.mean(abs_arr)),
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "abs_p95": float(np.percentile(abs_arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def profile_reward_method(
    *,
    method_source: str,
    log_path: str | Path,
    episode_data_dir: str | Path,
    args,
) -> Dict[str, Any]:
    episode_ids, preferred_episode, dispreferred_episode, selection_status = _select_profile_episodes(
        log_path,
        args,
    )
    if not episode_ids:
        raise RuntimeError("no episodes available for reward profiling")

    fn = _build_candidate_fn(method_source)
    ep_dir = Path(episode_data_dir)

    all_values: List[np.ndarray] = []
    episode_means: Dict[int, float] = {}
    missing_files: List[str] = []
    profiled_ids: List[int] = []

    for episode_id in episode_ids:
        mat_path = ep_dir / f"data_ep{int(episode_id)}.mat"
        if not mat_path.exists():
            missing_files.append(mat_path.name)
            continue
        rewards = _reward_values_on_episode(fn, mat_path, args)
        if rewards.size == 0:
            continue
        profiled_ids.append(int(episode_id))
        all_values.append(rewards)
        episode_means[int(episode_id)] = float(np.mean(rewards))

    if not all_values:
        missing_txt = ", ".join(missing_files) if missing_files else "no valid episode_data"
        raise RuntimeError(f"reward profiling could not load any episode_data files ({missing_txt})")

    merged = np.concatenate(all_values)
    summary = _summarize_values(merged)
    primary_delta = None
    if preferred_episode in episode_means and dispreferred_episode in episode_means:
        primary_delta = float(episode_means[preferred_episode] - episode_means[dispreferred_episode])

    return {
        "episode_ids": profiled_ids,
        "selection_status": selection_status,
        "preferred_episode": preferred_episode,
        "dispreferred_episode": dispreferred_episode,
        "preferred_avg_reward": episode_means.get(preferred_episode),
        "dispreferred_avg_reward": episode_means.get(dispreferred_episode),
        "primary_delta": primary_delta,
        "missing_files": missing_files,
        "episode_means": episode_means,
        **summary,
    }


def auto_tune_lrs_for_reward_patch(
    *,
    current_method_source: str,
    candidate_method_source: str,
    log_path: str | Path,
    episode_data_dir: str | Path,
    args,
    lr_proposal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_lr_reference_attrs(args)

    if not bool(getattr(args, "auto_lr_on_reward_patch", True)):
        return {"applied": False, "reason": "auto_lr_on_reward_patch=False"}
    if not bool(getattr(args, "reset_agent_on_patch", True)):
        return {"applied": False, "reason": "requires reset_agent_on_patch=True"}

    try:
        current_profile = profile_reward_method(
            method_source=current_method_source,
            log_path=log_path,
            episode_data_dir=episode_data_dir,
            args=args,
        )
        candidate_profile = profile_reward_method(
            method_source=candidate_method_source,
            log_path=log_path,
            episode_data_dir=episode_data_dir,
            args=args,
        )
    except Exception as exc:
        return {"applied": False, "reason": f"profiling_failed: {exc}"}

    scale_ratio = _safe_ratio(candidate_profile.get("abs_p95"), current_profile.get("abs_p95"), default=1.0)
    rms_ratio = _safe_ratio(candidate_profile.get("rms"), current_profile.get("rms"), default=1.0)
    delta_ratio = _safe_ratio(
        abs(candidate_profile.get("primary_delta") or 0.0),
        abs(current_profile.get("primary_delta") or 0.0),
        default=1.0,
    )
    trend_summary = _training_trend_summary(log_path, args)

    relative_factor = (scale_ratio ** -0.50) * (rms_ratio ** -0.25)
    relative_factor *= _clip(delta_ratio ** 0.10, 0.95, 1.05)
    relative_factor *= float(trend_summary["trend_factor"])

    floor = float(getattr(args, "lr_auto_scale_floor", 0.55))
    ceiling = max(floor, float(getattr(args, "lr_auto_scale_ceiling", 1.10)))
    current_factor = _clip(float(getattr(args, "lr_auto_factor", 1.0)), floor, ceiling)
    if lr_proposal:
        applied_factor = _clip(float(lr_proposal.get("lr_factor", current_factor)), floor, ceiling)
    else:
        applied_factor = _clip(current_factor * relative_factor, floor, ceiling)

    base_critic = float(getattr(args, "lr_critic_base"))
    base_critic_min = float(getattr(args, "lr_critic_min_base"))
    base_actor = float(getattr(args, "lr_actor_base"))
    base_actor_min = float(getattr(args, "lr_actor_min_base"))
    base_warmup = float(getattr(args, "lr_warmup_ratio_base"))

    warmup_gain = float(getattr(args, "lr_auto_warmup_gain", 0.10))
    warmup_min = float(getattr(args, "lr_auto_warmup_min", 0.10))
    warmup_max = float(getattr(args, "lr_auto_warmup_max", 0.20))
    if lr_proposal and "warmup_ratio" in lr_proposal:
        warmup_ratio = _clip(float(lr_proposal["warmup_ratio"]), warmup_min, warmup_max)
    else:
        warmup_ratio = _clip(
            base_warmup + warmup_gain * max(0.0, 1.0 - applied_factor),
            warmup_min,
            warmup_max,
        )

    args.lr_critic = float(base_critic * applied_factor)
    args.lr_critic_min = float(min(base_critic * applied_factor, base_critic_min * applied_factor))
    args.lr_actor = float(base_actor * applied_factor)
    args.lr_actor_min = float(min(base_actor * applied_factor, base_actor_min * applied_factor))
    args.lr_warmup_ratio = float(warmup_ratio)
    args.lr_auto_factor = float(applied_factor)

    return {
        "applied": True,
        "selection_status": candidate_profile["selection_status"],
        "profile_episode_ids": candidate_profile["episode_ids"],
        "current_profile": current_profile,
        "candidate_profile": candidate_profile,
        "scale_ratio": float(scale_ratio),
        "rms_ratio": float(rms_ratio),
        "delta_ratio": float(delta_ratio),
        "trend_summary": trend_summary,
        "relative_factor": float(relative_factor),
        "previous_factor": float(current_factor),
        "applied_factor": float(applied_factor),
        "lr_proposal_authoritative": bool(lr_proposal),
        "lr_proposal": lr_proposal or {},
        "applied_settings": {
            "lr_critic": float(args.lr_critic),
            "lr_critic_min": float(args.lr_critic_min),
            "lr_actor": float(args.lr_actor),
            "lr_actor_min": float(args.lr_actor_min),
            "lr_alpha": float(getattr(args, "lr_alpha", 0.0)),
            "lr_warmup_ratio": float(args.lr_warmup_ratio),
        },
    }
