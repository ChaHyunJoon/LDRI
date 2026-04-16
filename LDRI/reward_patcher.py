"""Safely patch only EMS.get_reward from LLM output."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
from typing import Iterable, List, Optional


class PatchError(RuntimeError):
    """Raised when extraction or patch validation fails."""


@dataclass
class PatchResult:
    target_file: str
    backup_file: Optional[str]
    class_name: str
    method_name: str
    old_method_source: str
    new_method_source: str

# LLM response 전체 텍스트에서 원하는 함수(get_reward)만 추출
def extract_method_from_llm_response(response_text: str, method_name: str = "get_reward") -> str:
    """Extract function source from fenced code blocks or raw response."""
    blocks = _extract_code_blocks(response_text)

    for code in blocks:
        src = _extract_function_source(code, method_name=method_name)
        if src is not None:
            return src

    src = _extract_function_source(response_text, method_name=method_name)
    if src is not None:
        return src

    raise PatchError(f"could not find function '{method_name}' in LLM response")

# 응답 텍스트 안에서 triple backtick 코드 블록만 추출
def _extract_code_blocks(text: str) -> List[str]:
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    return [m.group(1).strip("\n") for m in pattern.finditer(text)]

# 문자열 코드에서 특정 함수 이름의 source code만 AST 기반으로 추출함
def _extract_function_source(code: str, method_name: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    lines = code.splitlines()

    # top-level function
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            return textwrap.dedent(seg).strip("\n") + "\n"

    # method inside class
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    seg = "\n".join(lines[sub.lineno - 1 : sub.end_lineno])
                    return textwrap.dedent(seg).strip("\n") + "\n"

    return None

# patch 전에 함수 코드가 최소한의 문법, 이름, self 여부, 최소 key을 만족하는지 검사
def validate_method_source(
    method_source: str,
    method_name: str = "get_reward",
    required_info_keys: Optional[Iterable[str]] = None,
) -> None:
    """Validate syntax/signature/basic contract before patching."""
    wrapped = "class _Tmp:\n" + textwrap.indent(method_source.strip("\n") + "\n", "    ")
    try:
        tree = ast.parse(wrapped)
    except SyntaxError as e:
        raise PatchError(f"invalid method syntax: {e}") from e

    cls = tree.body[0]
    if not isinstance(cls, ast.ClassDef):
        raise PatchError("internal validation failed: wrapper class not parsed")

    funcs = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        raise PatchError("no function found in method source")
    fn = funcs[0]

    if fn.name != method_name:
        raise PatchError(f"expected method '{method_name}', got '{fn.name}'")

    if not fn.args.args or fn.args.args[0].arg != "self":
        raise PatchError("method signature must start with self")

    if required_info_keys:
        for key in required_info_keys:
            if f"'{key}'" not in method_source and f'"{key}"' not in method_source:
                raise PatchError(f"required info key '{key}' not found in candidate method")

# candidate reward를 EMS.get_reward에 투입
def patch_method_in_file(
    file_path: str | Path,
    method_source: str,
    class_name: str = "EMS",
    method_name: str = "get_reward",
    required_info_keys: Optional[Iterable[str]] = None,
    backup: bool = True,
    insert_if_missing: bool = False,
) -> PatchResult:
    """Patch only the given class method, keeping all other code untouched."""
    # 잘못된 코드는 애초에 patch 안 함
    validate_method_source(
        method_source,
        method_name=method_name,
        required_info_keys=required_info_keys,
    )

    path = Path(file_path)
    if not path.exists():
        raise PatchError(f"target file not found: {path}")
    # AST parse
    original = path.read_text(encoding="utf-8")
    tree = ast.parse(original)
    lines = original.splitlines(keepends=True)

    class_node = None
    method_node = None
    for node in tree.body:
        # EMS.get_reward를 찾는 과정
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_node = node
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    method_node = sub
                    break
            break

    if class_node is None:
        raise PatchError(f"class '{class_name}' not found in {path}")
    if method_node is None and not insert_if_missing:
        raise PatchError(f"method '{class_name}.{method_name}' not found in {path}")

    class_line = lines[class_node.lineno - 1]
    class_indent = class_line[: len(class_line) - len(class_line.lstrip())]
    method_indent = class_indent + "    "

    method_lines = textwrap.dedent(method_source).strip("\n").splitlines()
    new_block = [
        (method_indent + ln if ln.strip() else "") + "\n"
        for ln in method_lines
    ]

    if method_node is not None:
        # 기존 get_reward 있으면 잘라내고 새로운 block을 넣는다
        start = method_node.lineno - 1
        end = method_node.end_lineno
        old_method_source = "".join(lines[start:end])
    else:
        # Insert as a new class method at the end of the class block.
        start = class_node.end_lineno
        end = class_node.end_lineno
        old_method_source = ""

    patched_lines = lines[:start] + new_block + lines[end:]
    patched_text = "".join(patched_lines)

    # Syntax safety check before writing
    try:
        ast.parse(patched_text)
    except SyntaxError as e:
        raise PatchError(f"patched file syntax invalid: {e}") from e

    backup_path = None
    if backup:
        backup_path = str(path.with_suffix(path.suffix + ".bak"))
        Path(backup_path).write_text(original, encoding="utf-8")

    path.write_text(patched_text, encoding="utf-8")

    return PatchResult(
        target_file=str(path),
        backup_file=backup_path,
        class_name=class_name,
        method_name=method_name,
        old_method_source=old_method_source,
        new_method_source="\n".join(new_block),
    )

# 기존 get_reward와 새 get_reward의 difference를 unified diff 형식으로 보여줌
def method_diff_preview(old_method_source: str, new_method_source: str, max_lines: int = 120) -> str:
    import difflib

    old_lines = old_method_source.splitlines()
    new_lines = new_method_source.splitlines()
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old_get_reward",
            tofile="new_get_reward",
            lineterm="",
        )
    )
    if not diff:
        return "(no method changes)"

    if len(diff) > max_lines:
        shown = diff[:max_lines]
        shown.append(f"... ({len(diff) - max_lines} more diff lines omitted)")
        return "\n".join(shown)
    return "\n".join(diff)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Patch EMS.get_reward from LLM response text")
    parser.add_argument("--response-file", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--class-name", default="EMS")
    parser.add_argument("--method-name", default="get_reward")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    response_text = Path(args.response_file).read_text(encoding="utf-8")
    method = extract_method_from_llm_response(response_text, method_name=args.method_name)
    result = patch_method_in_file(
        file_path=args.target_file,
        method_source=method,
        class_name=args.class_name,
        method_name=args.method_name,
        required_info_keys=[
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
        ],
        backup=(not args.no_backup),
    )
    print(f"patched: {result.target_file}")
    if result.backup_file:
        print(f"backup: {result.backup_file}")
