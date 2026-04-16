# Codex Prompt for Generation of Initial Reward Design (FCEV-EMS, SAC)

## 1) System Instruction (generation stage)

You are designing `EMS.get_reward(self)` from scratch for FCEV energy management.

Target file and method:
- File: `FCHEV-EMS/LDRI/agentEMS_for_feedback.py`
- Class: `EMS`
- Method: `get_reward(self)`

Primary objective for this project:
1. Minimize hydrogen-related cost:
   - prefer lower `h2_fcs` and lower `h2_cost`
   - keep `h2_cost` tied to pure fuel-cell hydrogen consumption, not equivalent-hydrogen battery credit/debit
   - do not create fake gains by charging the battery above target just to make `h2_batt` more negative
2. Keep SOC near `self.SOC_target`:
   - default target is around `0.60`
   - keep SOC charge-sustaining and near target with simple causal tracking
   - do not overreact to small transient in-bounds deviation
3. Keep the reward simple enough for SAC to learn:
   - avoid over-layered SOC shaping
   - avoid very steep per-step penalties
4. Keep the additive objective fixed:
   - `objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost`
   - `reward = -objective_cost`

Hard constraints:
1. Edit only `EMS.get_reward(self)`.
2. Keep exactly four top-level cost components:
   - `soc_cost`
   - `h2_cost`
   - `fcs_soh_cost`
   - `batt_soh_cost`
3. Keep pure fuel-cell hydrogen accounting inside `h2_cost`:
   - `h2_cost = float(self.w_h2) * float(h2_fcs)`
   - if `h2_batt` or `h2_equal` is still computed for logging/analysis, it must not change `h2_cost`
4. Keep required keys in `self.info`:
   - `{REQUIRED_INFO_KEYS}`
5. Keep `soc_cost`, `fcs_soh_cost`, and `batt_soh_cost` non-negative and finite.
6. Return scalar `float`.
7. Do not use non-causal shaping:
   - no `step`
   - no episode progress
   - no remaining horizon
   - no terminal-only logic

SOC design rules:
1. Start from the minimum viable structure:
   - one base target-tracking term
   - one stronger out-of-bounds strengthening term
2. Allowed simple additions only if clearly needed:
   - a small deadband around `SOC_target`
   - a softer in-bounds slope
3. Do not stack multiple high-SOC or asymmetry terms by default.
4. Keep out-of-bounds pressure stronger than in-bounds pressure.
5. Use sign-safe distances:
   - `dist_low = max(0.0, soc_low - soc)`
   - `dist_high = max(0.0, soc - soc_high)`

TPE compatibility requirements:
1. Assume TPE replay provides only state-like signals:
   - `SOC`
   - `P_batt`
   - `P_FCS`
   - `h2_fcs`
   - `dSOH_FCS`
   - `dSOH_batt`
2. Therefore, the reward must remain causal and state-only.
3. The initial reward should make obviously worse hydrogen/SOC behavior receive lower average reward under replay too.

Output format:
1. `Analysis`
2. `Updated Code`
   - full `def get_reward(self): ...` only
3. `Expected Effects`
   - short bullets on hydrogen-related cost, SOC tracking, and learnability

## 2) Generation Prompt Template

Generate an initial reward function for RL training.

### Task Context
- RL algorithm: `{RL_ALGO}`
- Driving cycle(s): `{DRIVING_CYCLES}`
- Episode steps: `{EPISODE_STEPS}`
- Reward version: `{REWARD_VERSION}`
- SOC in-bounds floor reference for later TPE (%): `{TPE_SOC_FLOOR}`

### Environment Abstraction
```python
{ENVIRONMENT_DESCRIPTION}
```

### Task Description
{TASK_DESCRIPTION}

### Reward Template Contract
```python
{REWARD_TEMPLATE}
```

### Generation Requirements & Design Checklist
1. Keep objective decomposition explicit:
   - `objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost`
2. Build `h2_cost` from pure `h2_fcs`, not from `h2_equal` and not as a separate top-level term.
3. Make hydrogen reduction the first objective.
4. Keep SOC charge-sustaining near `0.60`.
5. Keep `soc_cost` simple:
   - base target tracking
   - out-of-bounds strengthening
   - optional small deadband only if needed
6. Avoid reward hacking from signed battery credit:
   - charging above target should not become attractive just because `h2_batt` is negative
   - battery-equivalent terms may be logged, but they must not reduce `h2_cost`
7. Avoid overly sharp or complicated penalties.
8. Keep the reward causal, stable, and easy to tune later.
9. Ensure code works in both training and smoke-test contexts.
10. Prefer simplicity over clever shaping.
