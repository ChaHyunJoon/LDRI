"""Independent LDRI entrypoint that uses simplified feasible-pool TPE.

This keeps the original workflow intact and swaps only:
1. feedback bundle builder
2. ldri_tpe module resolved inside the retry loop
3. runtime defaults suitable for a simple two-block TPE gate
"""

from __future__ import annotations

import sys

import main_ldri_refine as base
import ldri_tpe_feasible_pool as pool_tpe
from arguments_ldri import get_ldri_args as _get_ldri_args


_NEW_FILE_V = "v_ldri_feasible_pool"
_ACTIVE_ARGS = None


def _apply_feasible_pool_defaults(args):
    if str(getattr(args, "file_v", "") or "") == "v_ldri":
        args.file_v = _NEW_FILE_V

    args.workflow_variant = "ldri_feasible_pool"
    args.tpe_variant = "feasible_pool_simple_two_block"

    args.tpe_fp_pool_size = int(getattr(args, "tpe_fp_pool_size", 6))
    args.tpe_fp_topk = int(getattr(args, "tpe_fp_topk", 2))
    args.tpe_fp_bottomk = int(getattr(args, "tpe_fp_bottomk", 2))
    args.tpe_fp_guard_margin = float(getattr(args, "tpe_fp_guard_margin", 0.0))

    current_margin = float(getattr(args, "tpe_margin", 0.0))
    if abs(current_margin - 3.0) < 1e-12 or abs(current_margin) < 1e-12:
        args.tpe_margin = 0.05

    return args


def get_ldri_args():
    global _ACTIVE_ARGS
    args = _get_ldri_args()
    args = _apply_feasible_pool_defaults(args)
    _ACTIVE_ARGS = args
    pool_tpe.set_runtime_config_from_args(args)
    return args


def build_feedback_bundle(
    log_path,
    episode_data_dir=None,
    process_episodes=5,
    soc_bounds=(0.4, 0.8),
    preference_in_bounds_floor=0.0,
):
    if _ACTIVE_ARGS is not None:
        pool_tpe.set_runtime_config_from_args(_ACTIVE_ARGS)
    return pool_tpe.build_feedback_bundle_feasible_pool(
        log_path=log_path,
        episode_data_dir=episode_data_dir,
        process_episodes=process_episodes,
        soc_bounds=soc_bounds,
        preference_in_bounds_floor=preference_in_bounds_floor,
    )


def main():
    base.get_ldri_args = get_ldri_args
    base.build_feedback_bundle = build_feedback_bundle
    sys.modules["ldri_tpe"] = pool_tpe
    base.main()


if __name__ == "__main__":
    main()
