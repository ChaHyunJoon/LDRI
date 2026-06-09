# Prompt for Generation of Initial Reward Design (FCEV-EMS, SAC)

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
5. Keep the weighted cost skeleton explicit:
   - `soc_cost` must contain `float(self.w_soc) * float(soc - soc_target) ** 2`
   - `h2_cost = float(self.w_h2) * float(h2_fcs)`
   - `fcs_soh_cost = float(self.w_fc) * float(self.dSOH_FCS)`
   - `batt_soh_cost = float(self.w_batt) * float(self.dSOH_batt)`

Hard constraints:
1. Edit only `EMS.get_reward(self)`.
2. Keep exactly four top-level cost components:
   - `soc_cost`
   - `h2_cost`
   - `fcs_soh_cost`
   - `batt_soh_cost`
3. Keep the base weighted formulas present in the code:
   - `soc_cost` must contain `float(self.w_soc) * float(soc - soc_target) ** 2`
   - `h2_cost = float(self.w_h2) * float(h2_fcs)`
   - `fcs_soh_cost = float(self.w_fc) * float(self.dSOH_FCS)`
   - `batt_soh_cost = float(self.w_batt) * float(self.dSOH_batt)`
4. Keep pure fuel-cell hydrogen accounting inside `h2_cost`:
   - do not read `self.h2_batt` or `self.h2_equal` inside the reward computation logic
   - write `float(self.h2_batt)` and `float(self.h2_equal)` to `self.info` for logging only; do not use them as reward inputs
5. Do not replace degradation terms with SOH-state proxies:
   - do not use `(self.SOH_FCS - 1.0) ** 2` instead of `self.dSOH_FCS`
   - do not use `(self.SOH_batt - 1.0) ** 2` instead of `self.dSOH_batt`
6. Keep required keys in `self.info`:
   - `{REQUIRED_INFO_KEYS}`
7. `soc_cost` must be a continuous quadratic function — do not use step multipliers or `np.where`-based weight switching at SOC boundaries.
8. Keep `soc_cost`, `fcs_soh_cost`, and `batt_soh_cost` non-negative and finite.
9. Return scalar `float`.
10. Do not use non-causal shaping:
   - no `step`
   - no episode progress
   - no remaining horizon
   - no terminal-only logic

SOC design rules:
1. Start from the minimum viable weighted structure:
   - one base quadratic target-tracking term: `float(self.w_soc) * float(soc - soc_target) ** 2`
   - optionally one continuous quadratic out-of-bounds strengthening term using `dist_low` / `dist_high`, also scaled by `self.w_soc`
   - do not use step multipliers (e.g. `w_soc * 10`) that create reward discontinuities at SOC boundaries
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
3. For TPE/stub compatibility, read only `h2_fcs` for hydrogen logic.
4. Never read `self.h2_batt` or `self.h2_equal` inside reward computation; write `float(self.h2_batt)` and `float(self.h2_equal)` to `self.info` for logging only.
5. The initial reward should make obviously worse hydrogen/SOC behavior receive lower average reward under replay too.

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
2. Keep the base weighted skeleton explicit in code:
   - `soc_cost` anchored on `float(self.w_soc) * float(soc - soc_target) ** 2`
   - `h2_cost = float(self.w_h2) * float(h2_fcs)`
   - `fcs_soh_cost = float(self.w_fc) * float(self.dSOH_FCS)`
   - `batt_soh_cost = float(self.w_batt) * float(self.dSOH_batt)`
3. Build `h2_cost` from pure `h2_fcs`, not from `h2_equal` and not as a separate top-level term.
4. Use degradation increments directly:
   - use `self.dSOH_FCS` and `self.dSOH_batt`
   - do not substitute `(SOH - 1)^2` proxies
5. Make hydrogen reduction the first objective.
6. Keep SOC charge-sustaining near `0.60`.
7. Keep `soc_cost` as a continuous quadratic structure:
   - base quadratic tracking: `float(self.w_soc) * float(soc - soc_target) ** 2`
   - optional continuous out-of-bounds strengthening: add `self.w_soc * k * (dist_low ** 2 + dist_high ** 2)` with `k` between 2–5
   - do not use step multipliers or `np.where`-based weight switching at SOC boundaries
   - optional small deadband only if needed
8. Avoid reward hacking from signed battery credit:
   - charging above target should not become attractive just because `h2_batt` is negative
   - battery-equivalent terms are logged via `float(self.h2_batt)` / `float(self.h2_equal)` in `self.info` but must not be used as reward inputs
9. Avoid overly sharp or complicated penalties.
10. Keep the reward causal, stable, and easy to tune later.
11. Ensure code works in both training and smoke-test contexts.
12. Prefer simplicity over clever shaping.
