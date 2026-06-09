"""Observability helpers for LDRI reward-refinement workflow.
observability: 관찰 가능성, logging: 기록, 
"""

from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

# 지정된 경로에 json 형식으로 데이터를 저장하는 함수
def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# 지정된 경로에 txt 형식으로 데이터를 저장하는 함수
def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# LLM 응답에서 analysis 섹션만 분리하는 함수
def extract_analysis_section(response_text: str) -> str:
    """Extract analysis rationale from LLM response for easy review."""
    # Look for an explicit "Analysis" section (case-insensitive, allowing for optional colon and whitespace).
    heading = re.search(r"(?im)^\s*analysis\s*:?\s*$", response_text)
    if heading:
        start = heading.end()
        after = response_text[start:]
        code_pos = after.find("```")
        # analysis head가 있으면 첫 code block 전까지의 텍스트를 반환, 없으면 첫 code block 전까지의 텍스트를 반환
        if code_pos >= 0:
            body = after[:code_pos]
        # 명시적인 analysis 섹션이 없다면 응답 텍스트의 처음부터 첫 번째 코드 블록까지의 텍스트를 반환합니다.
        else:
            body = after
        body = body.strip()
        if body:
            return body

    # Fallback: everything before the first code block.
    code_pos = response_text.find("```")
    if code_pos >= 0:
        head = response_text[:code_pos].strip()
        if head:
            return head

    return "(analysis section not explicitly found)"

# 
def _num_from_node(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sub = _num_from_node(node.operand)
        return None if sub is None else -sub
    return None


def _collect_numeric_assignments(method_source: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    try:
        tree = ast.parse(method_source)
    except SyntaxError:
        return values

    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn = node
            break
    if fn is None:
        return values

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            num = _num_from_node(node.value)
            if num is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = num
    return values

# weight or penalty가 포함된 변수명 중에서 숫자 할당이 있는 경우, 그 변수명과 값의 변화를 추적하는 함수
def infer_key_weight_changes(old_method: str, new_method: str) -> List[Dict[str, Any]]:
    """Capture rough scalar weight/penalty/scale changes for leaderboard view."""
    old = _collect_numeric_assignments(old_method)
    new = _collect_numeric_assignments(new_method)

    hints = ("weight", "penalty", "scale", "multiplier", "coef", "alpha", "beta")
    names = sorted(set(old.keys()) | set(new.keys()))
    out = []
    for name in names:
        if not any(h in name.lower() for h in hints):
            continue
        ov = old.get(name)
        nv = new.get(name)
        if ov is None and nv is None:
            continue
        if ov == nv:
            continue
        delta = None if (ov is None or nv is None) else (nv - ov)
        out.append({"name": name, "old": ov, "new": nv, "delta": delta})
    return out

# tpe logging: 여러 reward candidate를 평가하고, 어느 시도가 parse가 되었고, 어떤 reward가 더 선호되고, margin이 얼마가 났는지
def leaderboard_markdown(attempts: List[Dict[str, Any]]) -> str:
    lines = [
        "| attempt | parse_ok | tpe_pass | preferred_avg | dispreferred_avg | margin | delta | note |",
        "|---:|:---:|:---:|---:|---:|---:|---:|---|",
    ]
    for a in attempts:
        lines.append(
            "| {attempt} | {parse_ok} | {tpe_pass} | {preferred_avg} | {dispreferred_avg} | {margin} | {delta} | {note} |".format(
                attempt=a.get("attempt", ""),
                parse_ok=str(a.get("parse_ok", "")),
                tpe_pass=str(a.get("tpe_pass", "")),
                preferred_avg=_fmt_num(a.get("preferred_avg")),
                dispreferred_avg=_fmt_num(a.get("dispreferred_avg")),
                margin=_fmt_num(a.get("margin")),
                delta=_fmt_num(a.get("delta")),
                note=(a.get("note", "") or "").replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt_num(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.6f}"
    except Exception:
        return str(v)

# 한 iteration에 대한 상태, 시도, 결과를 누적 기록하는 observer 객체
class LDRIObserver:
    def __init__(self, iter_dir: Path, iteration: int):
        self.iter_dir = Path(iter_dir)
        self.iteration = int(iteration)
        self.summary_path = self.iter_dir / "iteration_summary.json"
        self.leaderboard_path = self.iter_dir / "candidate_leaderboard.md"
        self.summary: Dict[str, Any] = {
            "iteration": self.iteration,
            "status_flow": [],
            "attempts": [],
            "outcome": {},
        }
        self.persist()
    # 현재 iteration의 상태 기록
    def status(self, status: str, detail: Optional[str] = None) -> None:
        rec = {
            "time": dt.datetime.now().isoformat(),
            "status": status,
            "detail": detail,
        }
        self.summary["status_flow"].append(rec)
        msg = f"[iter {self.iteration:03d}] STATUS: {status}"
        if detail:
            msg += f" | {detail}"
        print(msg)
        self.persist()
    # candidate reward의 attempt 결과를 summary에 추가
    def add_attempt(self, attempt_payload: Dict[str, Any]) -> None:
        self.summary["attempts"].append(attempt_payload)
        self.persist()
        _write_text(self.leaderboard_path, leaderboard_markdown(self.summary["attempts"]))
    # iteration의 최종 결과를 저장함.
    def set_outcome(self, **kwargs: Any) -> None:
        self.summary["outcome"].update(kwargs)
        self.persist()

    def persist(self) -> None:
        _write_json(self.summary_path, self.summary)

