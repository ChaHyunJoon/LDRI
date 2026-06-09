from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "llama": AgentPreset(
        agent_type="llama",
        provider="huggingface",
        model="meta-llama/Llama-3.1-70B-Instruct",
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


# Per-agent model assignment per provider.
# trainability_scheduler is excluded: replaced by pure Python computation.
AGENT_MODELS: Dict[str, Dict[str, str]] = {
    "anthropic": {
        # Stage 2a: specialist reviewers
        # soc_safety / fuel_efficiency: structured JSON, rule-heavy → Haiku sufficient
        # fc_durability / battery_usage: structured JSON, fuel-priority persona → Haiku sufficient
        # reward_hacking: code structure analysis → Sonnet for reliability
        "soc_safety":        "claude-haiku-4-5-20251001",
        "fuel_efficiency":   "claude-haiku-4-5-20251001",
        "fc_durability":     "claude-haiku-4-5-20251001",
        "battery_usage":     "claude-haiku-4-5-20251001",
        "reward_hacking":    "claude-sonnet-4-6",
        # Stage 2.5: trade-off mediator — conflict resolution requires reasoning
        "tradeoff_mediator": "claude-sonnet-4-6",
        # Stage 3: orchestrator — synthesizes all reviewer + mediator outputs → best model
        "orchestrator":      "claude-opus-4-7",
        # Stage 4: proposer — generates actual Python reward code
        "proposer":          "claude-sonnet-4-6",
    },
    "openai": {
        "soc_safety":        "gpt-4.1-mini",
        "fuel_efficiency":   "gpt-4.1-mini",
        "fc_durability":     "gpt-4.1-mini",
        "battery_usage":     "gpt-4.1-mini",
        "reward_hacking":    "gpt-4.1",
        "tradeoff_mediator": "gpt-4.1",
        "orchestrator":      "gpt-5.4",
        "proposer":          "gpt-4.1",
    },
    "huggingface": {
        "soc_safety":        "meta-llama/Llama-3.1-70B-Instruct",
        "fuel_efficiency":   "meta-llama/Llama-3.1-70B-Instruct",
        "fc_durability":     "meta-llama/Llama-3.1-70B-Instruct",
        "battery_usage":     "meta-llama/Llama-3.1-70B-Instruct",
        "reward_hacking":    "meta-llama/Llama-3.1-70B-Instruct",
        "tradeoff_mediator": "meta-llama/Llama-3.1-70B-Instruct",
        "orchestrator":      "meta-llama/Llama-3.1-70B-Instruct",
        "proposer":          "meta-llama/Llama-3.1-70B-Instruct",
    },
}


def get_agent_model(provider: str, agent_name: str, fallback_model: str) -> str:
    """Return the per-agent model for a given provider.

    Falls back to fallback_model when the provider or agent_name is not in AGENT_MODELS.
    """
    provider_map = AGENT_MODELS.get(str(provider or "").lower().strip(), {})
    return provider_map.get(agent_name, fallback_model)
