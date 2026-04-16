"""End-to-end orchestration loop for reward refinement.

Flow:
1) (optional) run training command
2) parse logs / episode_data into process+trajectory+preference feedback
3) render prompt from markdown template
4) call LLM
5) extract and patch EMS.get_reward only
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import subprocess
from typing import Dict

from feedback_parser import LogParseError, build_feedback_bundle
from llm_client import LLMClient, LLMClientError, get_api_key
from prompt_builder import (
    PromptBuildError,
    build_iteration_prompt,
    extract_method_source,
)
from reward_patcher import (
    PatchError,
    extract_method_from_llm_response,
    method_diff_preview,
    patch_method_in_file,
)


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


def _run_training(train_cmd: str, cwd: str | None = None) -> None:
    print(f"[train] running: {train_cmd}")
    result = subprocess.run(train_cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"training command failed with exit code {result.returncode}")


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_replacements(args: argparse.Namespace, bundle: Dict[str, str], current_method: str, version: str) -> Dict[str, str]:
    return {
        "RL_ALGO": args.rl_algo,
        "DRIVING_CYCLES": args.driving_cycles,
        "EPISODE_STEPS": str(args.episode_steps),
        "REWARD_VERSION": version,
        "CURRENT_GET_REWARD_CODE": current_method.strip("\n"),
        "PROCESS_FEEDBACK": bundle["PROCESS_FEEDBACK"],
        "TRAJECTORY_FEEDBACK": bundle["TRAJECTORY_FEEDBACK"],
        "PREFERENCE_FEEDBACK": bundle["PREFERENCE_FEEDBACK"],
    }


def run_iteration(args: argparse.Namespace, iteration: int, client: LLMClient | None) -> None:
    print(f"\n=== Iteration {iteration} ===")

    if args.train_cmd and not args.skip_train:
        _run_training(args.train_cmd, cwd=args.train_cwd)
    else:
        print("[train] skipped")

    try:
        bundle = build_feedback_bundle(
            log_path=args.log_path,
            episode_data_dir=args.episode_data_dir,
            process_episodes=args.process_episodes,
            soc_bounds=(args.soc_low, args.soc_high),
            preference_in_bounds_floor=getattr(args, "tpe_soc_in_bounds_floor", 0.0),
        )
    except LogParseError as e:
        raise RuntimeError(f"feedback parsing failed: {e}") from e

    current_method = extract_method_source(
        file_path=args.target_file,
        class_name=args.class_name,
        method_name=args.method_name,
    )

    version = f"{args.reward_version_prefix}_{iteration}"
    replacements = _build_replacements(args, bundle, current_method, version)
    prompt = build_iteration_prompt(args.template_path, replacements)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_text(out_dir / f"iter_{iteration:02d}_prompt.md", prompt)

    if args.dry_run_llm:
        print("[llm] dry-run enabled; prompt generated only")
        return

    if client is None:
        raise RuntimeError("LLM client not initialized")

    response = client.complete(
        prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
    )
    _save_text(out_dir / f"iter_{iteration:02d}_response.md", response)

    new_method = extract_method_from_llm_response(response, method_name=args.method_name)
    diff_preview = method_diff_preview(current_method, new_method)
    _save_text(out_dir / f"iter_{iteration:02d}_method.diff", diff_preview)
    print("[patch] method diff preview saved")

    if args.no_apply:
        print("[patch] --no-apply enabled; file patch skipped")
        return

    patch_result = patch_method_in_file(
        file_path=args.target_file,
        method_source=new_method,
        class_name=args.class_name,
        method_name=args.method_name,
        required_info_keys=REQUIRED_INFO_KEYS,
        backup=(not args.no_backup),
    )

    print(f"[patch] patched file: {patch_result.target_file}")
    if patch_result.backup_file:
        print(f"[patch] backup: {patch_result.backup_file}")

    # Lightweight syntax check for target file
    subprocess.run(["python3", "-m", "py_compile", args.target_file], check=True)
    print("[check] py_compile passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-driven reward refinement orchestration loop")

    parser.add_argument("--iterations", type=int, default=1)

    parser.add_argument("--template-path", default="./reward_refinement_codex_prompt.md")
    parser.add_argument("--target-file", default="./agentEMS_for_feedback.py")
    parser.add_argument("--class-name", default="EMS")
    parser.add_argument("--method-name", default="get_reward")

    parser.add_argument("--log-path", required=True)
    parser.add_argument("--episode-data-dir", default=None)
    parser.add_argument("--process-episodes", type=int, default=5)
    parser.add_argument("--soc-low", type=float, default=0.4)
    parser.add_argument("--soc-high", type=float, default=0.8)

    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--timeout-sec", type=int, default=180)

    parser.add_argument("--train-cmd", default=None)
    parser.add_argument("--train-cwd", default=None)
    parser.add_argument("--skip-train", action="store_true")

    parser.add_argument("--rl-algo", default="SAC")
    parser.add_argument("--driving-cycles", default="UNKNOWN")
    parser.add_argument("--episode-steps", default="UNKNOWN")
    parser.add_argument("--reward-version-prefix", default="auto_refine")

    parser.add_argument("--output-dir", default="./llm_refine_outputs")
    parser.add_argument("--dry-run-llm", action="store_true")
    parser.add_argument("--no-apply", action="store_true")
    parser.add_argument("--no-backup", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve relative paths from current working directory
    args.template_path = str(Path(args.template_path).resolve())
    args.target_file = str(Path(args.target_file).resolve())
    args.log_path = str(Path(args.log_path).resolve())
    if args.episode_data_dir:
        args.episode_data_dir = str(Path(args.episode_data_dir).resolve())
    if args.train_cwd:
        args.train_cwd = str(Path(args.train_cwd).resolve())

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = str(Path(args.output_dir).resolve() / run_id)

    client = None
    if not args.dry_run_llm:
        api_key = get_api_key(args.provider, args.api_key, args.api_key_env)
        client = LLMClient(provider=args.provider, model=args.model, api_key=api_key)

    for i in range(1, args.iterations + 1):
        run_iteration(args, i, client)

    print(f"\nDone. Artifacts saved in: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, PromptBuildError, PatchError, LLMClientError) as e:
        raise SystemExit(f"[ERROR] {e}")
