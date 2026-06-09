"""Default TPE entrypoint mapped to the simplified two-block feasible-pool gate.

Primary block:
- choose one preferred/dispreferred pair from SOC-qualified episodes
- selector priority: lower objective_cost, then lower h2+degradation cost,
  then smaller final SOC deviation
- hard gate: preferred_avg_reward > dispreferred_avg_reward + margin

Additional block:
- compare top feasible pool mean reward vs bottom feasible pool mean reward
- diagnostic only, not a hard reject condition
"""

from __future__ import annotations

from pathlib import Path

from ldri_tpe_feasible_pool import (
    TPEResult,
    build_feedback_bundle_feasible_pool,
    evaluate_tpe_candidate,
    set_runtime_config_from_args,
)

__all__ = [
    "TPEResult",
    "build_feedback_bundle_feasible_pool",
    "evaluate_tpe_candidate",
    "set_runtime_config_from_args",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run simplified two-block TPE on candidate get_reward"
    )
    parser.add_argument(
        "--method-file",
        required=True,
        help="file containing top-level get_reward function code",
    )
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
            process_episodes=5,
            soc_bounds=(args.soc_low, args.soc_high),
            preference_in_bounds_floor=args.tpe_soc_in_bounds_floor,
        )
        print("\n[PROCESS_FEEDBACK]\n")
        print(bundle["PROCESS_FEEDBACK"])
        print("\n[PREFERENCE_FEEDBACK]\n")
        print(bundle["PREFERENCE_FEEDBACK"])
        print("\n[TRAJECTORY_FEEDBACK]\n")
        print(bundle["TRAJECTORY_FEEDBACK"])
