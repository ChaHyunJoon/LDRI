# ARROW: Automated Reward Refinement with Orchestrated Workflow

ARROW is an LLM-driven reward iteration framework for training a Deep Reinforcement Learning (DRL)-based Energy Management System (EMS) for Fuel Cell Electric Vehicles (FCEV). Instead of manually designing reward functions, ARROW automatically refines them iteration by iteration by combining SAC training feedback with LLM-based multi-agent review.

---

## Overview

Training a DRL agent for FCEV-EMS requires a carefully shaped reward function that balances hydrogen consumption, SOC regulation, fuel cell durability, and battery degradation. ARROW automates this process:

1. **Bootstrap** — An LLM generates an initial `get_reward()` function from scratch using a task description and environment specification.
2. **Train** — The SAC agent trains for a fixed number of episodes (`chunk_episodes`) using the current reward.
3. **Review** — A multi-agent pipeline analyzes the training log and proposes an improved reward.
4. **Gate** — The proposed reward is validated by an alignment gate before being applied.
5. **Repeat** — Steps 2–4 repeat for `ldri_iterations` iterations.

---

## Repository Structure

```
.
├── LDRI/                          # Core ARROW workflow modules
│   ├── main_ldri_refine.py        # Entry point — outer LDRI loop
│   ├── runner_ldri.py             # SAC training executor per iteration
│   ├── arguments_ldri.py          # All CLI arguments
│   ├── agentEMS_for_feedback.py   # Reward module (patched each iteration)
│   ├── multi_agent_review.py      # 6-stage multi-agent review pipeline
│   ├── feedback_parser.py         # Parses training logs into feedback bundles
│   ├── prompt_builder.py          # Builds LLM prompts from feedback + history
│   ├── llm_client.py              # LLM API client (OpenAI / Anthropic / HuggingFace)
│   ├── llm_presets.py             # Per-agent model assignments per provider
│   ├── reward_patcher.py          # Patches get_reward() in the reward module
│   ├── ldri_aligned_gate.py       # Two-layer alignment gate (Kendall τ + dominance)
│   ├── ldri_observer.py           # Logs analysis, diffs, and key weight changes
│   ├── reward_lr_tuner.py         # Auto-tunes LR bands after each reward patch
│   ├── ems_tools.py               # Python tool executor for agentic tool use
│   ├── reward_generation_prompt.md  # Bootstrap prompt template
│   └── reward_refinement_prompt.md  # Iteration prompt template
├── common/                        # Shared RL utilities
│   ├── env.py                     # FCEV-EMS Gym environment
│   ├── network.py                 # SAC actor/critic networks
│   ├── memory.py                  # Replay buffer
│   ├── runner.py                  # Episode runner
│   ├── evaluate.py                # Evaluation utilities
│   └── utils.py                   # Misc helpers
├── main.py                        # Baseline SAC training (no ARROW)
├── sac.py                         # SAC algorithm implementation
├── agentEMS.py                    # Base EMS reward module
├── Battery.py                     # Battery model
├── DP_env.py                      # Dynamic programming environment
└── ...
```

---

## Multi-Agent Review Pipeline

When `--multi_agent_review True`, the reward refinement uses a 6-stage pipeline instead of a single LLM call:

```
context_builder (Python)
    └─► diagnostic_agent (Python)
    └─► [Specialist Reviewers — LLM × 5]
          ├─ soc_safety
          ├─ fuel_efficiency       ← PRIMARY OBJECTIVE
          ├─ reward_hacking
          ├─ fc_durability
          └─ battery_usage
    └─► scale_reviewer (Python)
    └─► trainability_scheduler (Python)
    └─► tradeoff_mediator (LLM × 1)
    └─► orchestrator (LLM × 1)
    └─► counterfactual_evaluator (Python)
    └─► proposer (LLM × 1..N)
    └─► TPE / aligned-gate verifier (Python)
```

**Fuel efficiency (g H₂/km) is always the primary objective.** SOC is treated as a constraint that enables fuel efficiency, not an end in itself.

### Per-Agent Model Assignment (Anthropic)

| Agent | Model |
|---|---|
| soc_safety, fuel_efficiency, fc_durability, battery_usage | claude-haiku-4-5 |
| reward_hacking, tradeoff_mediator, proposer | claude-sonnet-4-6 |
| orchestrator | claude-opus-4-7 |

---

## Aligned Gate

Before applying a proposed reward patch, ARROW runs a two-layer alignment gate:

- **Layer 1 (Kendall τ):** The candidate reward's episode rankings must correlate with physical performance rankings (H₂/100km + SOC in-bounds rate). Threshold: `--aligned_min_tac` (default 0.3).
- **Layer 2 (Dominance check):** No single cost term may dominate the objective. SOC cost is capped at `--aligned_soc_dominance_threshold` (default 0.85) and H₂ cost must contribute at least `--aligned_min_h2_fraction` (default 0.25).

A patch that fails either layer is rejected and the current reward is kept.

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy scipy tqdm anthropic openai
```

### 2. Set API key

```bash
export ANTHROPIC_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here
```

### 3. Run ARROW

```bash
cd LDRI
python main_ldri_refine.py \
    --agent_type claude \
    --scenario_name Standard_WLTC_WVUINTER \
    --MODE CS \
    --soc0 0.6 \
    --ldri_iterations 5 \
    --chunk_episodes 400 \
    --feedback_process_episodes 10 \
    --multi_agent_review True
```

## Output Structure

Each run creates a timestamped folder under `--ldri_root` (default `./ldri_runs/`):

```
ldri_runs/<timestamp>_<scenario>/
├── run_config.json          # Full run configuration snapshot
├── run_summary.json         # Final results summary
├── bootstrap/               # Initial reward generation artifacts
│   ├── candidate_get_reward_attempt_1.py
│   ├── bootstrap_summary.json
│   └── ...
├── iter_001/                # Per-iteration artifacts
│   ├── train_log.log
│   ├── candidate_get_reward_attempt_1.py
│   ├── best_candidate_vs_current.diff
│   ├── iteration_summary.json
│   └── ...
├── iter_002/
└── models/
    └── iter_001/<scenario>/net_params/best_meta_iter001.json
```
