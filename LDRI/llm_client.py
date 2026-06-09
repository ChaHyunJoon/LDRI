"""Minimal provider client for OpenAI/Anthropic/Hugging Face using stdlib HTTP.
외부 SDK 없이 urllib만 써서 prompt를 보내고 completion text를 받아오는 역할 (문자열 prompt 넣고 문자열 응답 받는...)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
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


@dataclass
class LLMCompletion:
    text: str
    provider: str
    model: str
    usage: Dict[str, Any]
    cost: Dict[str, Any]


class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str):
        # Normalize provider once so downstream dispatch is deterministic.
        self.provider = provider.lower().strip()
        self.model = model
        self.api_key = api_key
        if self.provider not in {"openai", "anthropic", "huggingface"}:
            raise LLMClientError(f"unsupported provider: {provider}")
    # prompt별로 최종 text return. model을 지정하면 해당 호출에만 client.model을 override함.
    def complete(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout_sec: int = 120,
        model: Optional[str] = None,
    ) -> str:
        req = LLMRequest(
            provider=self.provider,
            model=model if model is not None else self.model,
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
        if req.provider == "huggingface":
            return _huggingface_complete(req)
        raise LLMClientError(f"unsupported provider: {req.provider}")

    def complete_with_tools(
        self,
        prompt: str,
        tools: list,
        tool_executor,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout_sec: int = 120,
        model: Optional[str] = None,
        max_tool_rounds: int = 8,
    ) -> LLMCompletion:
        """Call LLM with Anthropic tool_use support. Falls back to complete_with_usage for other providers.

        tool_executor: callable(tool_name: str, tool_input: dict) -> str
        Returns LLMCompletion with cumulative usage across all tool rounds.
        """
        effective_model = _normalize_model_name(
            self.provider, model if model is not None else self.model
        )
        if self.provider == "anthropic" and tools:
            req = LLMRequest(
                provider=self.provider,
                model=effective_model,
                prompt=prompt,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
            return _anthropic_complete_with_tools(req, tools, tool_executor, max_tool_rounds)
        return self.complete_with_usage(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            model=model,
        )

    def complete_with_usage(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout_sec: int = 120,
        model: Optional[str] = None,
    ) -> LLMCompletion:
        req = LLMRequest(
            provider=self.provider,
            model=model if model is not None else self.model,
            prompt=prompt,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        if req.provider == "openai":
            return _openai_complete_with_usage(req)
        if req.provider == "anthropic":
            return _anthropic_complete_with_usage(req)
        if req.provider == "huggingface":
            return _huggingface_complete_with_usage(req)
        raise LLMClientError(f"unsupported provider: {req.provider}")


DEFAULT_HTTP_USER_AGENT = "FCEV-EMS-LDRI/1.0 (Python urllib)"
ANTHROPIC_MODEL_ALIASES = {
    "claude-sonnet-4.6": "claude-sonnet-4-6",
}
DEFAULT_PRICE_USD_PER_1M = {
    "anthropic:claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "openai:gpt-4.1": {"input": 2.0, "output": 8.0},
}


def get_api_key(provider: str, explicit_key: Optional[str], api_key_env: Optional[str]) -> str:
    # Priority: CLI explicit key > custom env var > provider default env var.
    if explicit_key:
        return explicit_key

    provider = provider.lower().strip()
    if api_key_env:
        key = os.getenv(api_key_env)
        if key:
            return key

    fallback_env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "huggingface": "HF_TOKEN",
    }
    fallback_env = fallback_env_map.get(provider, "OPENAI_API_KEY")
    key = os.getenv(fallback_env)
    if key:
        return key

    raise LLMClientError(
        f"API key not found. Set --api-key, or environment variable "
        f"{api_key_env or fallback_env}."
    )


def _normalize_model_name(provider: str, model: str) -> str:
    provider = str(provider or "").strip().lower()
    model = str(model or "").strip()
    if provider == "anthropic":
        return ANTHROPIC_MODEL_ALIASES.get(model, model)
    return model

# JSON body를 HTTP POST로 전송, 응답도 JSON으로 parse, HTTP/network 예외를 통일해서 처리(HTTPError, URLError)
# OpenAI와 Anthropic 코드에서 HTTP 처리 로직을 중복하지 않음.
def _http_post_json(url: str, headers: dict, body: dict, timeout_sec: int) -> dict:
    # Single JSON POST helper used by all providers to keep error handling uniform.
    payload = json.dumps(body).encode("utf-8")
    merged_headers = dict(headers)
    merged_headers.setdefault("User-Agent", DEFAULT_HTTP_USER_AGENT)
    req = request.Request(url=url, data=payload, headers=merged_headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return json.loads(text)
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise LLMClientError(f"HTTP {e.code} from provider: {detail}") from e
    except error.URLError as e:
        raise LLMClientError(f"network error: {e}") from e


def _is_temperature_rejected_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "temperature" in text
        and (
            "unsupported parameter" in text
            or "deprecated" in text
            or "not supported" in text
            or "does not support" in text
        )
    )


def _usage_cost(provider: str, model: str, usage: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate USD cost from usage and optional per-1M-token price config.

    Configure prices with:
      LLM_PRICE_USD_PER_1M_JSON='{"anthropic:claude-x":{"input":3,"output":15}}'

    Keys may be "provider:model" or just "model". Rates are deliberately not
    hard-coded here because provider pricing changes independently of the repo.
    """
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)

    price_table = DEFAULT_PRICE_USD_PER_1M
    price_table_raw = os.getenv("LLM_PRICE_USD_PER_1M_JSON", "").strip()
    price_entry = None
    if price_table_raw:
        try:
            price_table = json.loads(price_table_raw)
        except Exception:
            price_table = DEFAULT_PRICE_USD_PER_1M

    price_entry = (
        price_table.get(f"{provider}:{model}")
        or price_table.get(model)
        or price_table.get(f"{provider}:*")
    )

    input_cost = None
    output_cost = None
    total_cost = None
    if isinstance(price_entry, dict):
        in_rate = price_entry.get("input")
        out_rate = price_entry.get("output")
        if input_tokens is not None and in_rate is not None:
            input_cost = float(input_tokens) * float(in_rate) / 1_000_000.0
        if output_tokens is not None and out_rate is not None:
            output_cost = float(output_tokens) * float(out_rate) / 1_000_000.0
        if input_cost is not None or output_cost is not None:
            total_cost = float(input_cost or 0.0) + float(output_cost or 0.0)

    return {
        "currency": "USD",
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "pricing_source": "LLM_PRICE_USD_PER_1M_JSON" if price_table_raw and price_entry else "code_default" if price_entry else None,
        "input_usd_per_1m_tokens": price_entry.get("input") if isinstance(price_entry, dict) else None,
        "output_usd_per_1m_tokens": price_entry.get("output") if isinstance(price_entry, dict) else None,
        "total_tokens": total_tokens,
    }


def _completion_from_data(req: LLMRequest, text: str, data: Dict[str, Any]) -> LLMCompletion:
    model = _normalize_model_name(req.provider, data.get("model") or req.model)
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    usage = dict(usage)
    if "total_tokens" not in usage:
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        if input_tokens is not None and output_tokens is not None:
            usage["total_tokens"] = int(input_tokens) + int(output_tokens)
    return LLMCompletion(
        text=text,
        provider=req.provider,
        model=model,
        usage=usage,
        cost=_usage_cost(req.provider, model, usage),
    )


def _openai_complete(req: LLMRequest) -> str:
    return _openai_complete_with_usage(req).text


def _openai_complete_with_usage(req: LLMRequest) -> LLMCompletion:
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
        if not _is_temperature_rejected_error(e):
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
    return _completion_from_data(req, text, data)


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
    return _anthropic_complete_with_usage(req).text


def _anthropic_complete_with_usage(req: LLMRequest) -> LLMCompletion:
    # Anthropic message response may include mixed content blocks; keep text blocks only.
    url = "https://api.anthropic.com/v1/messages"
    model = _normalize_model_name(req.provider, req.model)
    headers = {
        "x-api-key": req.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": req.max_tokens,
        "messages": [{"role": "user", "content": req.prompt}],
    }
    if req.temperature is not None:
        body["temperature"] = req.temperature

    try:
        data = _http_post_json(url, headers, body, req.timeout_sec)
    except LLMClientError as e:
        # Newer Anthropic models may reject temperature entirely.
        if not _is_temperature_rejected_error(e):
            raise
        body.pop("temperature", None)
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
    return _completion_from_data(req, text, data)


def _anthropic_complete_with_tools(
    req: LLMRequest,
    tools: list,
    tool_executor,
    max_tool_rounds: int,
) -> LLMCompletion:
    """Anthropic tool_use loop: call API, execute tools, repeat until end_turn."""
    url = "https://api.anthropic.com/v1/messages"
    model = _normalize_model_name(req.provider, req.model)
    headers = {
        "x-api-key": req.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    messages = [{"role": "user", "content": req.prompt}]
    cumulative_input_tokens = 0
    cumulative_output_tokens = 0
    final_text = ""

    for _round in range(max_tool_rounds + 1):
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "tools": tools,
            "messages": messages,
        }
        if req.temperature is not None:
            body["temperature"] = req.temperature

        try:
            data = _http_post_json(url, headers, body, req.timeout_sec)
        except LLMClientError as e:
            if not _is_temperature_rejected_error(e):
                raise
            body.pop("temperature", None)
            data = _http_post_json(url, headers, body, req.timeout_sec)

        usage = data.get("usage", {})
        cumulative_input_tokens  += int(usage.get("input_tokens",  0))
        cumulative_output_tokens += int(usage.get("output_tokens", 0))

        stop_reason = data.get("stop_reason", "end_turn")
        content     = data.get("content", [])

        # Collect any text blocks from this response
        text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
        if text_parts:
            final_text = "\n".join(text_parts).strip()

        if stop_reason != "tool_use":
            break

        # Execute all tool_use blocks and build tool_result user message
        messages.append({"role": "assistant", "content": content})
        tool_results = []
        for block in content:
            if block.get("type") != "tool_use":
                continue
            result_str = tool_executor(block["name"], block.get("input", {}))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": result_str,
            })
        if not tool_results:
            break
        messages.append({"role": "user", "content": tool_results})

    if not final_text:
        raise LLMClientError("Anthropic tool_use response did not contain text output")

    cumulative_usage = {
        "input_tokens":  cumulative_input_tokens,
        "output_tokens": cumulative_output_tokens,
        "total_tokens":  cumulative_input_tokens + cumulative_output_tokens,
    }
    return LLMCompletion(
        text=final_text,
        provider=req.provider,
        model=model,
        usage=cumulative_usage,
        cost=_usage_cost(req.provider, model, cumulative_usage),
    )


def _extract_chat_completions_text(data: dict) -> str:
    out = []
    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            out.append(content.strip())
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
    return "\n".join(out).strip()


def _has_hf_routing_suffix(model: str) -> bool:
    return ":" in str(model or "")


def _hf_preferred_model(model: str) -> str:
    if _has_hf_routing_suffix(model):
        return model
    return f"{model}:preferred"


def _should_retry_hf_with_preferred(model: str, err: Exception) -> bool:
    if _has_hf_routing_suffix(model):
        return False
    text = str(err)
    return (
        "HTTP 403" in text
        or "HTTP 429" in text
        or "HTTP 5" in text
        or "network error" in text
    )


def _huggingface_complete(req: LLMRequest) -> str:
    return _huggingface_complete_with_usage(req).text


def _huggingface_complete_with_usage(req: LLMRequest) -> LLMCompletion:
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {req.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": req.model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
    }
    if req.temperature is not None:
        body["temperature"] = req.temperature

    try:
        data = _http_post_json(url, headers, body, req.timeout_sec)
    except LLMClientError as e:
        if not _should_retry_hf_with_preferred(req.model, e):
            raise
        body["model"] = _hf_preferred_model(req.model)
        data = _http_post_json(url, headers, body, req.timeout_sec)

    text = _extract_chat_completions_text(data)
    if not text:
        raise LLMClientError(
            "Hugging Face response did not contain text output. "
            f"summary={_compact_json({'model': data.get('model'), 'id': data.get('id')})}"
        )
    return _completion_from_data(req, text, data)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick LLM completion test")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "huggingface"],
        required=True,
    )
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
