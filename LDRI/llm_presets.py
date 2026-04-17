from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentPreset:
    agent_type: str
    provider: str
    model: str
    api_key_env: str


AGENT_PRESETS = {
    "gpt": AgentPreset(
        agent_type="gpt",
        provider="openai",
        model="gpt-5.4",
        api_key_env="OPENAI_API_KEY",
    ),
    "claude": AgentPreset(
        agent_type="claude",
        provider="anthropic",
        model="claude-sonnet-4.6",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "llama": AgentPreset(
        agent_type="llama",
        provider="huggingface",
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key_env="HF_TOKEN",
    ),
}


def get_agent_type_choices():
    return tuple(AGENT_PRESETS.keys())


def get_agent_preset(agent_type: str) -> AgentPreset:
    normalized = str(agent_type or "gpt").strip().lower()
    try:
        return AGENT_PRESETS[normalized]
    except KeyError as exc:
        valid = ", ".join(get_agent_type_choices())
        raise ValueError(f"unsupported agent_type: {agent_type!r}. Expected one of: {valid}") from exc


def apply_agent_preset(args):
    preset = get_agent_preset(getattr(args, "agent_type", "gpt"))
    args.agent_type = preset.agent_type
    if not getattr(args, "llm_provider", None):
        args.llm_provider = preset.provider
    if not getattr(args, "llm_model", None):
        args.llm_model = preset.model
    if not getattr(args, "llm_api_key_env", None):
        args.llm_api_key_env = preset.api_key_env
    return args
