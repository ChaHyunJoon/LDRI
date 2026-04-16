"""Minimal provider client for OpenAI/Anthropic using stdlib HTTP.
외부 SDK 없이 urllib만 써서 prompt를 보내고 completion text를 받아오는 역할 (문자열 prompt 넣고 문자열 응답 받는...)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional
from urllib import error, request


class LLMClientError(RuntimeError):
    """Raised when provider request fails."""


@dataclass
# provider / model / api key를 받아 client 생성
class LLMRequest:
    provider: str
    model: str
    prompt: str
    api_key: str
    temperature: float = 0.2
    max_tokens: int = 2000
    timeout_sec: int = 120


class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str):
        # Normalize provider once so downstream dispatch is deterministic.
        self.provider = provider.lower().strip()
        self.model = model
        self.api_key = api_key
        if self.provider not in {"openai", "anthropic"}:
            raise LLMClientError(f"unsupported provider: {provider}")
    # prompt별로 최종 text return
    def complete(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout_sec: int = 120,
    ) -> str:
        req = LLMRequest(
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        if req.provider == "openai":
            return _openai_complete(req)
        if req.provider == "anthropic":
            return _anthropic_complete(req)
        raise LLMClientError(f"unsupported provider: {req.provider}")


def get_api_key(provider: str, explicit_key: Optional[str], api_key_env: Optional[str]) -> str:
    # Priority: CLI explicit key > custom env var > provider default env var.
    if explicit_key:
        return explicit_key

    provider = provider.lower().strip()
    if api_key_env:
        key = os.getenv(api_key_env)
        if key:
            return key

    fallback_env = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    key = os.getenv(fallback_env)
    if key:
        return key

    raise LLMClientError(
        f"API key not found. Set --api-key, or environment variable "
        f"{api_key_env or fallback_env}."
    )

# JSON body를 HTTP POST로 전송, 응답도 JSON으로 parse, HTTP/network 예외를 통일해서 처리(HTTPError, URLError)
# OpenAI와 Anthropic 코드에서 HTTP 처리 로직을 중복하지 않음.
def _http_post_json(url: str, headers: dict, body: dict, timeout_sec: int) -> dict:
    # Single JSON POST helper used by all providers to keep error handling uniform.
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(url=url, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return json.loads(text)
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise LLMClientError(f"HTTP {e.code} from provider: {detail}") from e
    except error.URLError as e:
        raise LLMClientError(f"network error: {e}") from e


def _openai_complete(req: LLMRequest) -> str:
    # Responses API returns structured output; _extract_openai_text flattens it.
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {req.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": req.model,
        "input": req.prompt,
        "max_output_tokens": req.max_tokens,
    }
    if req.temperature is not None:
        body["temperature"] = req.temperature

    try:
        data = _http_post_json(url, headers, body, req.timeout_sec)
    except LLMClientError as e:
        # Some OpenAI models reject temperature; retry once without it.
        if "Unsupported parameter: 'temperature'" not in str(e):
            raise
        body.pop("temperature", None)
        data = _http_post_json(url, headers, body, req.timeout_sec)

    text = _extract_openai_text(data)
    if not text:
        if data.get("error"):
            raise LLMClientError(
                f"OpenAI response error payload: {_compact_json(data.get('error'))}"
            )
        if data.get("incomplete_details"):
            raise LLMClientError(
                "OpenAI response incomplete: "
                f"{_compact_json(data.get('incomplete_details'))}"
            )
        raise LLMClientError(
            "OpenAI response did not contain text output. "
            f"status={data.get('status')!r}, summary={_compact_json(_response_summary(data))}"
        )
    return text


def _extract_openai_text(data: dict) -> str:
    # Fast path for SDK-like condensed field.
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    if isinstance(output_text, list):
        out = []
        for item in output_text:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t.strip():
                    out.append(t.strip())
        if out:
            return "\n".join(out).strip()

    # Fallback for nested tool/content format.
    out = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue

        # Some payloads place text/refusal at item-level.
        for key in ("text", "refusal"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())

        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            t = content.get("text")
            if isinstance(t, str) and t.strip():
                out.append(t.strip())

            # Refusal blocks may use {"type":"refusal","refusal":"..."}.
            refusal = content.get("refusal")
            if isinstance(refusal, str) and refusal.strip():
                out.append(refusal.strip())

            # Reasoning blocks may include nested summary text entries.
            for summary in content.get("summary", []):
                if not isinstance(summary, dict):
                    continue
                st = summary.get("text")
                if isinstance(st, str) and st.strip():
                    out.append(st.strip())

    # Compatibility fallback if backend returns chat-completions-like payload.
    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            out.append(content.strip())
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                for key in ("text", "value", "content"):
                    value = part.get(key)
                    if isinstance(value, str) and value.strip():
                        out.append(value.strip())
                        break
    return "\n".join(out).strip()


def _compact_json(value, max_len: int = 500) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _response_summary(data: dict) -> dict:
    output = data.get("output")
    output_items = []
    if isinstance(output, list):
        for item in output[:6]:
            if not isinstance(item, dict):
                output_items.append(type(item).__name__)
                continue
            info = {"type": item.get("type"), "status": item.get("status")}
            content = item.get("content")
            if isinstance(content, list):
                info["content_types"] = [
                    c.get("type") for c in content[:6] if isinstance(c, dict)
                ]
            output_items.append(info)

    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "status": data.get("status"),
        "output_items": output_items,
        "has_output_text": isinstance(data.get("output_text"), str),
    }


def _anthropic_complete(req: LLMRequest) -> str:
    # Anthropic message response may include mixed content blocks; keep text blocks only.
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": req.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "messages": [{"role": "user", "content": req.prompt}],
    }
    data = _http_post_json(url, headers, body, req.timeout_sec)

    out = []
    for part in data.get("content", []):
        if part.get("type") == "text":
            txt = part.get("text")
            if isinstance(txt, str) and txt:
                out.append(txt)
    text = "\n".join(out).strip()
    if not text:
        raise LLMClientError("Anthropic response did not contain text output")
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick LLM completion test")
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()

    key = get_api_key(args.provider, args.api_key, args.api_key_env)
    client = LLMClient(provider=args.provider, model=args.model, api_key=key)
    print(client.complete(args.prompt, temperature=args.temperature, max_tokens=args.max_tokens))
