"""Simplified feasible-pool TPE gate for LDRI reward refinement.

This version intentionally keeps the TPE logic small:
1. Primary block:
   - choose one preferred/dispreferred pair from SOC-qualified episodes
   - selector priority: lower objective_cost, then lower h2+degradation cost,
     then smaller final SOC deviation
   - hard gate: preferred_avg_reward > dispreferred_avg_reward + margin
2. Additional block:
   - compare top feasible pool mean reward vs bottom feasible pool mean reward
   - diagnostic only, not a hard reject condition

The goal is to avoid over-gating reward updates while still enforcing one clear
pairwise ranking condition that matches the task proxy used in training logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as scio

from feedback_parser import (
    EpisodeSummary,
    _summarize_trajectory,
    format_process_feedback,
    parse_train_log,
)


@dataclass
class TPEResult:
    passed: bool
    preferred_episode: Optional[int]
    dispreferred_episode: Optional[int]
    preferred_avg_reward: Optional[float]
    dispreferred_avg_reward: Optional[float]
    margin: float
    message: str


@dataclass
class FeasiblePoolSelection:
    best: EpisodeSummary
    degraded: EpisodeSummary
    top_pool: List[EpisodeSummary]
    bottom_pool: List[EpisodeSummary]
    status: str


_DEFAULT_RUNTIME_CONFIG: Dict[str, float] = {
    "soc_target": 0.6,
    "tpe_soc_in_bounds_floor": 100.0,
    "tpe_fp_pool_size": 6.0,
    "tpe_fp_topk": 2.0,
    "tpe_fp_bottomk": 2.0,
    "tpe_fp_guard_margin": 0.0,
}

_runtime_config: Dict[str, float] = dict(_DEFAULT_RUNTIME_CONFIG)


def set_runtime_config_from_args(args) -> None:
    global _runtime_config
    cfg = dict(_DEFAULT_RUNTIME_CONFIG)
    for key in cfg:
        if hasattr(args, key):
            try:
                cfg[key] = float(getattr(args, key))
            except Exception:
                pass
    _runtime_config = cfg


def _cfg_value(name: str, args=None) -> float:
    if args is not None and hasattr(args, name):
        try:
            return float(getattr(args, name))
        except Exception:
            pass
    return float(_runtime_config.get(name, _DEFAULT_RUNTIME_CONFIG[name]))


def _pool_size(args=None) -> int:
    return max(4, int(round(_cfg_value("tpe_fp_pool_size", args=args))))


def _topk(args=None) -> int:
    return max(1, int(round(_cfg_value("tpe_fp_topk", args=args))))


def _bottomk(args=None) -> int:
    return max(1, int(round(_cfg_value("tpe_fp_bottomk", args=args))))


def _episode_objective_metric(ep: EpisodeSummary) -> Optional[float]:
    if ep.mean_objective_cost is not None:
        return float(ep.mean_objective_cost)
    if ep.objective_100km is not None:
        return float(ep.objective_100km)
    return None


def _episode_h2_degradation_metric(ep: EpisodeSummary) -> Optional[float]:
    if (
        ep.mean_h2_cost is not None
        and ep.mean_fcs_cost is not None
        and ep.mean_batt_cost is not None
    ):
        return float(ep.mean_h2_cost + ep.mean_fcs_cost + ep.mean_batt_cost)
    return None


def _episode_final_soc_dev(ep: EpisodeSummary, soc_target: float) -> Optional[float]:
    if ep.soc_end is None:
        return None
    return float(abs(float(ep.soc_end) - float(soc_target)))


def _episode_valid(
    ep: EpisodeSummary,
    *,
    min_in_bounds_rate: float,
    soc_target: float,
) -> bool:
    if ep.in_bounds_rate is None or float(ep.in_bounds_rate) < float(min_in_bounds_rate):
        return False
    if _episode_objective_metric(ep) is None:
        return False
    if _episode_h2_degradation_metric(ep) is None:
        return False
    if _episode_final_soc_dev(ep, soc_target) is None:
        return False
    return True


def _selection_rank_key(ep: EpisodeSummary, *, soc_target: float) -> Tuple[float, float, float, int]:
    objective_metric = _episode_objective_metric(ep)
    h2_metric = _episode_h2_degradation_metric(ep)
    final_dev = _episode_final_soc_dev(ep, soc_target)
    if objective_metric is None or h2_metric is None or final_dev is None:
        raise ValueError("episode is missing objective/h2/final-SOC metrics")
    return (
        float(objective_metric),
        float(h2_metric),
        float(final_dev),
        int(ep.order),
    )


def select_feasible_pool(
    episodes: Iterable[EpisodeSummary],
    *,
    soc_target: float,
    min_in_bounds_rate: float,
    pool_size: int,
    topk: int,
    bottomk: int,
) -> Tuple[Optional[FeasiblePoolSelection], str]:
    valid = [
        ep for ep in episodes
        if _episode_valid(
            ep,
            min_in_bounds_rate=min_in_bounds_rate,
            soc_target=soc_target,
        )
    ]
    if len(valid) < 2:
        return None, "not enough SOC-qualified episodes with objective_cost and h2+degradation metrics"

    ranked = sorted(valid, key=lambda ep: _selection_rank_key(ep, soc_target=soc_target))
    best = ranked[0]
    degraded = ranked[-1]
    if int(best.episode) == int(degraded.episode) and int(best.order) == int(degraded.order):
        return None, "preferred/dispreferred collapsed to same episode"

    half = max(1, min(len(ranked) // 2, int(pool_size) // 2))
    top_count = min(max(1, int(topk)), half)
    bottom_count = min(max(1, int(bottomk)), half)
    top_pool = ranked[:top_count]
    bottom_pool = ranked[-bottom_count:]

    status = (
        "ok (selector=objective_h2_simple; "
        f"in_bounds_floor>={float(min_in_bounds_rate):.1f}%; "
        f"preferred=ep{int(best.episode)}, dispreferred=ep{int(degraded.episode)}, "
        f"topk={len(top_pool)}, bottomk={len(bottom_pool)})"
    )
    return (
        FeasiblePoolSelection(
            best=best,
            degraded=degraded,
            top_pool=top_pool,
            bottom_pool=bottom_pool,
            status=status,
        ),
        status,
    )


def _episode_ref(ep: EpisodeSummary) -> str:
    return f"ep {int(ep.episode)}"


def _metric_summary(ep: EpisodeSummary, *, soc_target: float) -> str:
    parts = []
    if ep.in_bounds_rate is not None:
        parts.append(f"in_bounds_rate={float(ep.in_bounds_rate):.1f}%")
    obj = _episode_objective_metric(ep)
    if obj is not None:
        parts.append(f"mean_objective_cost={float(obj):.4f}")
    h2_metric = _episode_h2_degradation_metric(ep)
    if h2_metric is not None:
        parts.append(f"h2_plus_degradation_cost={float(h2_metric):.4f}")
    final_dev = _episode_final_soc_dev(ep, soc_target)
    if final_dev is not None:
        parts.append(f"final_soc_dev={float(final_dev):.4f}")
    if ep.soc_min is not None and ep.soc_mean is not None and ep.soc_max is not None:
        parts.append(
            "SOC[min/mean/max]={:.3f}/{:.3f}/{:.3f}".format(
                float(ep.soc_min),
                float(ep.soc_mean),
                float(ep.soc_max),
            )
        )
    if ep.ep_cumulative_r is not None:
        parts.append(f"ep_cumulative_r={float(ep.ep_cumulative_r):.3f}")
    return ", ".join(parts)


def _format_process_feedback_feasible_pool(
    episodes: List[EpisodeSummary],
    *,
    num_episodes: int,
    selection: Optional[FeasiblePoolSelection],
    soc_target: float,
) -> str:
    base = format_process_feedback(episodes, num_episodes=num_episodes)
    if selection is None:
        return base

    lines = [
        "",
        "[Simple TPE anchor episodes]",
        f"- primary preferred: {_episode_ref(selection.best)}: {_metric_summary(selection.best, soc_target=soc_target)}",
        f"- primary dispreferred: {_episode_ref(selection.degraded)}: {_metric_summary(selection.degraded, soc_target=soc_target)}",
        "- additional top pool: " + ", ".join(_episode_ref(ep) for ep in selection.top_pool),
        "- additional bottom pool: " + ", ".join(_episode_ref(ep) for ep in selection.bottom_pool),
    ]
    return base + "\n" + "\n".join(lines)


def format_preference_feedback_feasible_pool(
    episodes: List[EpisodeSummary],
    *,
    soc_target: float,
    min_in_bounds_rate: float,
    pool_size: int,
    topk: int,
    bottomk: int,
) -> str:
    selection, status = select_feasible_pool(
        episodes,
        soc_target=soc_target,
        min_in_bounds_rate=min_in_bounds_rate,
        pool_size=pool_size,
        topk=topk,
        bottomk=bottomk,
    )
    if selection is None:
        return f"N/A (simple feasible-pool preference feedback unavailable: {status})"

    lines = [
        "Preferred trajectory characteristics:",
        f"0) simple TPE selector: {status}",
        f"1) all compared episodes must satisfy in_bounds_rate >= {float(min_in_bounds_rate):.1f}%",
        "2) primary block: lower objective_cost is better",
        "3) if objective_cost is similar, lower h2_cost + degradation costs is better",
        "4) if both episodes are still similar, smaller final SOC deviation is better",
        "5) additional block: top feasible pool should receive higher mean reward than bottom feasible pool",
        "",
        "Primary block anchors:",
        f"- preferred ep {selection.best.episode}: {_metric_summary(selection.best, soc_target=soc_target)}",
        f"- dispreferred ep {selection.degraded.episode}: {_metric_summary(selection.degraded, soc_target=soc_target)}",
        "",
        "Additional block pools:",
        "- top feasible pool: " + ", ".join(_episode_ref(ep) for ep in selection.top_pool),
        "- bottom feasible pool: " + ", ".join(_episode_ref(ep) for ep in selection.bottom_pool),
        "",
        "Current violation example:",
        "- reward should assign higher average step reward to the primary preferred episode than to the primary dispreferred episode.",
        "- additional block is diagnostic only: top feasible pool should still score better on average than bottom feasible pool.",
    ]
    return "\n".join(lines)


def format_trajectory_feedback_feasible_pool(
    episodes: List[EpisodeSummary],
    episode_data_dir: Optional[str | Path],
    *,
    soc_bounds: Tuple[float, float],
    soc_target: float,
    min_in_bounds_rate: float,
    pool_size: int,
    topk: int,
    bottomk: int,
) -> str:
    if not episode_data_dir:
        return "N/A (episode_data directory not provided for trajectory-level analysis)"

    ep_dir = Path(episode_data_dir)
    if not ep_dir.exists():
        return f"N/A (episode_data directory not found: {ep_dir})"

    selection, status = select_feasible_pool(
        episodes,
        soc_target=soc_target,
        min_in_bounds_rate=min_in_bounds_rate,
        pool_size=pool_size,
        topk=topk,
        bottomk=bottomk,
    )
    if selection is None:
        return f"N/A (simple feasible-pool trajectory feedback unavailable: {status})"

    anchor_eps = [selection.best, selection.degraded]
    missing = []
    for ep in anchor_eps:
        mat_path = ep_dir / f"data_ep{int(ep.episode)}.mat"
        if not mat_path.exists():
            missing.append(mat_path.name)
    if missing:
        return "N/A (missing episode_data .mat files for selected simple TPE anchors: " + ", ".join(missing) + ")"

    lines = [
        "[Simple TPE trajectory snippets]",
        f"Selection rule: {status}",
        "",
        "[Primary preferred trajectory snippets]",
        _summarize_trajectory(
            selection.best.episode,
            ep_dir / f"data_ep{int(selection.best.episode)}.mat",
            "primary preferred trajectory",
            selection.best.ep_cumulative_r,
            soc_bounds,
        ),
        "",
        "[Primary dispreferred trajectory snippets]",
        _summarize_trajectory(
            selection.degraded.episode,
            ep_dir / f"data_ep{int(selection.degraded.episode)}.mat",
            "primary dispreferred trajectory",
            selection.degraded.ep_cumulative_r,
            soc_bounds,
        ),
    ]
    return "\n".join(lines)


def build_feedback_bundle_feasible_pool(
    log_path: str | Path,
    episode_data_dir: Optional[str | Path] = None,
    process_episodes: int = 5,
    soc_bounds: Tuple[float, float] = (0.4, 0.8),
    preference_in_bounds_floor: float = 0.0,
) -> Dict[str, str]:
    episodes = parse_train_log(log_path)
    soc_target = _cfg_value("soc_target")
    pool_size = _pool_size()
    topk = _topk()
    bottomk = _bottomk()
    selection, _ = select_feasible_pool(
        episodes,
        soc_target=soc_target,
        min_in_bounds_rate=preference_in_bounds_floor,
        pool_size=pool_size,
        topk=topk,
        bottomk=bottomk,
    )

    return {
        "PROCESS_FEEDBACK": _format_process_feedback_feasible_pool(
            episodes,
            num_episodes=process_episodes,
            selection=selection,
            soc_target=soc_target,
        ),
        "TRAJECTORY_FEEDBACK": format_trajectory_feedback_feasible_pool(
            episodes,
            episode_data_dir=episode_data_dir,
            soc_bounds=soc_bounds,
            soc_target=soc_target,
            min_in_bounds_rate=preference_in_bounds_floor,
            pool_size=pool_size,
            topk=topk,
            bottomk=bottomk,
        ),
        "PREFERENCE_FEEDBACK": format_preference_feedback_feasible_pool(
            episodes,
            soc_target=soc_target,
            min_in_bounds_rate=preference_in_bounds_floor,
            pool_size=pool_size,
            topk=topk,
            bottomk=bottomk,
        ),
    }


def _arr(mat, *keys, default_len=None):
    for k in keys:
        if k in mat:
            arr = np.asarray(mat[k], dtype=np.float64).squeeze()
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
        raise RuntimeError("candidate method source does not define get_reward")
    return fn


class _StubEMS:
    pass


def _average_candidate_reward_on_episode(fn, mat_path: Path, args) -> float:
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

    rewards = []
    for i in range(n):
        stub.SOC = float(soc[i])
        stub.h2_fcs = float(h2_fcs[i])
        stub.P_batt = float(p_batt_kw[i]) * 1000.0
        stub.P_FCS = float(p_fc_kw[i])
        stub.P_fc = stub.P_FCS
        stub.dSOH_FCS = float(dsoh_fcs[i])
        stub.dSOH_batt = float(dsoh_batt[i])
        stub.info = {}

        val = fn(stub)
        if isinstance(val, tuple):
            val = val[0]
        rewards.append(float(val))
    return float(np.mean(rewards))


def _average_for_episode(fn, ep_dir: Path, episode_id: int, args) -> float:
    mat_path = ep_dir / f"data_ep{int(episode_id)}.mat"
    if not mat_path.exists():
        raise RuntimeError(f"missing episode_data file: {mat_path.name}")
    return _average_candidate_reward_on_episode(fn, mat_path, args)


def evaluate_tpe_candidate(
    method_source: str,
    log_path: str | Path,
    episode_data_dir: str | Path,
    args,
    margin: float = 0.0,
) -> TPEResult:
    episodes = parse_train_log(log_path)
    soc_target = float(getattr(args, "soc_target", _cfg_value("soc_target", args=args)))
    in_bounds_floor = float(
        getattr(args, "tpe_soc_in_bounds_floor", _cfg_value("tpe_soc_in_bounds_floor", args=args))
    )
    pool_size = _pool_size(args=args)
    topk = _topk(args=args)
    bottomk = _bottomk(args=args)
    additional_margin = float(
        getattr(args, "tpe_fp_guard_margin", _cfg_value("tpe_fp_guard_margin", args=args))
    )
    ep_dir = Path(episode_data_dir)

    selection, status = select_feasible_pool(
        episodes,
        soc_target=soc_target,
        min_in_bounds_rate=in_bounds_floor,
        pool_size=pool_size,
        topk=topk,
        bottomk=bottomk,
    )
    if selection is None:
        return TPEResult(
            passed=False,
            preferred_episode=None,
            dispreferred_episode=None,
            preferred_avg_reward=None,
            dispreferred_avg_reward=None,
            margin=float(margin),
            message=f"Simple feasible-pool TPE skipped: {status}",
        )

    unique_eps: Dict[int, EpisodeSummary] = {}
    for ep in list(selection.top_pool) + list(selection.bottom_pool) + [selection.best, selection.degraded]:
        unique_eps[int(ep.episode)] = ep

    try:
        fn = _build_candidate_fn(method_source)
        reward_by_episode = {
            episode_id: _average_for_episode(fn, ep_dir, episode_id, args)
            for episode_id in unique_eps
        }
    except Exception as e:
        return TPEResult(
            passed=False,
            preferred_episode=int(selection.best.episode),
            dispreferred_episode=int(selection.degraded.episode),
            preferred_avg_reward=None,
            dispreferred_avg_reward=None,
            margin=float(margin),
            message=f"Simple feasible-pool TPE execution error: {e}",
        )

    best_avg = float(reward_by_episode[int(selection.best.episode)])
    degraded_avg = float(reward_by_episode[int(selection.degraded.episode)])
    primary_delta = float(best_avg - degraded_avg)
    primary_pass = bool(best_avg > (degraded_avg + float(margin)))

    top_mean = float(np.mean([reward_by_episode[int(ep.episode)] for ep in selection.top_pool]))
    bottom_mean = float(np.mean([reward_by_episode[int(ep.episode)] for ep in selection.bottom_pool]))
    additional_delta = float(top_mean - bottom_mean)
    additional_pass = bool(top_mean > (bottom_mean + float(additional_margin)))

    passed = bool(primary_pass)
    msg = (
        f"Simple feasible-pool TPE {'PASSED' if passed else 'FAILED'}: "
        f"primary_block=ep{int(selection.best.episode)}>ep{int(selection.degraded.episode)}, "
        f"primary_margin={float(margin):.6f}, primary_delta={primary_delta:.6f}; "
        f"additional_block=top_pool_vs_bottom_pool, "
        f"additional_margin={float(additional_margin):.6f}, additional_delta={additional_delta:.6f}, "
        f"top_mean={top_mean:.6f}, bottom_mean={bottom_mean:.6f}, "
        f"additional_result={'PASS' if additional_pass else 'FAIL'} (diagnostic only), "
        f"selector={selection.status}"
    )

    return TPEResult(
        passed=passed,
        preferred_episode=int(selection.best.episode),
        dispreferred_episode=int(selection.degraded.episode),
        preferred_avg_reward=float(best_avg),
        dispreferred_avg_reward=float(degraded_avg),
        margin=float(margin),
        message=msg,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run simplified feasible-pool TPE on candidate get_reward")
    parser.add_argument("--method-file", required=True, help="file containing top-level get_reward function code")
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
    parser.add_argument("--tpe-soc-in-bounds-floor", type=float, default=100.0)
    parser.add_argument("--tpe-fp-pool-size", type=int, default=6)
    parser.add_argument("--tpe-fp-topk", type=int, default=2)
    parser.add_argument("--tpe-fp-bottomk", type=int, default=2)
    parser.add_argument("--tpe-fp-guard-margin", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--print-feedback", action="store_true")
    args = parser.parse_args()

    set_runtime_config_from_args(args)
    method_source = Path(args.method_file).read_text(encoding="utf-8")
    result = evaluate_tpe_candidate(
        method_source=method_source,
        log_path=args.log_path,
        episode_data_dir=args.episode_data_dir,
        args=args,
        margin=args.margin,
    )
    print(result)

    if args.print_feedback:
        bundle = build_feedback_bundle_feasible_pool(
            log_path=args.log_path,
            episode_data_dir=args.episode_data_dir,
            preference_in_bounds_floor=args.tpe_soc_in_bounds_floor,
        )
        print("\n--- PROCESS_FEEDBACK ---\n")
        print(bundle["PROCESS_FEEDBACK"])
        print("\n--- TRAJECTORY_FEEDBACK ---\n")
        print(bundle["TRAJECTORY_FEEDBACK"])
        print("\n--- PREFERENCE_FEEDBACK ---\n")
        print(bundle["PREFERENCE_FEEDBACK"])
