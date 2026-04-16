"""Prompt template loader and placeholder builder for reward refinement."""

from __future__ import annotations

import ast
from pathlib import Path
import textwrap
from typing import Dict


RUNTIME_STOP_PREFIXES = (
    "## 3)",
    "## 3 ",
    "## 4)",
)


class PromptBuildError(RuntimeError):
    """Raised when prompt template extraction/substitution fails."""


def load_template(template_path: str | Path) -> str:
    path = Path(template_path)
    if not path.exists():
        raise PromptBuildError(f"template file not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_runtime_template(template_text: str) -> str:
    """Use only the system+iteration sections (drop usage guide/examples)."""
    lines = template_text.splitlines()
    out = []
    for line in lines:
        if any(line.startswith(prefix) for prefix in RUNTIME_STOP_PREFIXES):
            break
        out.append(line)
    runtime = "\n".join(out).strip()
    if not runtime:
        raise PromptBuildError("runtime prompt section is empty")
    return runtime + "\n"

#EMS에서 get_reward method를 추출하기 위한 함수
def extract_method_source(
    file_path: str | Path,
    class_name: str = "EMS",
    method_name: str = "get_reward",
) -> str:
    path = Path(file_path)
    if not path.exists():
        raise PromptBuildError(f"target file not found: {path}")

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    seg = "\n".join(lines[sub.lineno - 1 : sub.end_lineno])
                    return textwrap.dedent(seg).strip("\n") + "\n"

    raise PromptBuildError(
        f"method '{class_name}.{method_name}' not found in {path}"
    )


def fill_placeholders(template_text: str, mapping: Dict[str, str], strict: bool = True) -> str:
    rendered = template_text
    for key, value in mapping.items():
        rendered = rendered.replace("{" + key + "}", value)

    if strict:
        missing = []
        for key in ("RL_ALGO", "DRIVING_CYCLES", "EPISODE_STEPS", "REWARD_VERSION"):
            token = "{" + key + "}"
            if token in rendered:
                missing.append(token)
        if missing:
            raise PromptBuildError("unfilled core placeholders: " + ", ".join(missing))

    return rendered

#prompt에 들어가야할 정보를 입력해서 prompt를 완성하기
def build_iteration_prompt(
    template_path: str | Path,
    replacements: Dict[str, str],
) -> str:
    full_template = load_template(template_path)
    runtime_template = extract_runtime_template(full_template)
    return fill_placeholders(runtime_template, replacements, strict=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render runtime prompt from markdown template")
    parser.add_argument("--template-path", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--class-name", default="EMS")
    parser.add_argument("--method-name", default="get_reward")
    parser.add_argument("--rl-algo", default="SAC")
    parser.add_argument("--driving-cycles", default="UNKNOWN")
    parser.add_argument("--episode-steps", default="UNKNOWN")
    parser.add_argument("--reward-version", default="manual")
    parser.add_argument("--tpe-soc-floor", default="20.0")
    parser.add_argument("--process-feedback", default="N/A")
    parser.add_argument("--trajectory-feedback", default="N/A")
    parser.add_argument("--preference-feedback", default="N/A")
    args = parser.parse_args()

    method_code = extract_method_source(
        file_path=args.target_file,
        class_name=args.class_name,
        method_name=args.method_name,
    )

    prompt = build_iteration_prompt(
        template_path=args.template_path,
        replacements={
            "RL_ALGO": args.rl_algo,
            "DRIVING_CYCLES": args.driving_cycles,
            "EPISODE_STEPS": args.episode_steps,
            "REWARD_VERSION": args.reward_version,
            "TPE_SOC_FLOOR": str(args.tpe_soc_floor),
            "CURRENT_GET_REWARD_CODE": method_code.strip("\n"),
            "PROCESS_FEEDBACK": args.process_feedback,
            "TRAJECTORY_FEEDBACK": args.trajectory_feedback,
            "PREFERENCE_FEEDBACK": args.preference_feedback,
        },
    )
    print(prompt)
