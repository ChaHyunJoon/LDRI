# Prompt for Generation of Initial Reward Design (FCEV-EMS, SAC)

## 1) System Instruction (generation stage)

You are designing `EMS.get_reward(self)` from scratch for FCEV energy management.

Target file and method:
- File: `FCHEV-EMS/LDRI/agentEMS_for_feedback.py`
- Class: `EMS`
- Method: `get_reward(self)`

Economic cost model (this is the core intent — every term is real money in KRW):
- The weights are real-world market prices, NOT abstract tuning knobs:
  - `w_h2` ≈ 10 (KRW per gram of hydrogen — national H2 station average price)
  - `w_fc` ≈ 4.275e6 (KRW — full PEMFC stack replacement cost; multiplied by per-step `dSOH_FCS`)
  - `w_batt` ≈ 1.116e6 (KRW — full Li-ion pack replacement cost; multiplied by per-step `dSOH_batt`)
  - `w_soc` is the penalty price for SOC deviation from target / leaving safe bounds
- `objective_cost` is therefore the TOTAL per-step operating + degradation cost in KRW, and the
  agent's goal is to minimize this total monetary cost over the driving cycle.
- Hydrogen fuel is the dominant operating cost; FC and battery degradation are priced at replacement cost.

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
   - `soc_cost` base — for the INITIAL reward use the baseline-intent form (DEFAULT):
     - (a) pure quadratic with NO deadband (db=0): `float(self.w_soc) * float(soc - soc_target) ** 2` — tight target tracking
     - (b) deadband quadratic `float(self.w_soc) * k_inbounds * max(0.0, abs(soc - soc_target) - db) ** 2` is ALLOWED but is a lever for the LATER refinement stage to explore; do NOT use it as the initial choice
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
   - `soc_cost` base (initial reward DEFAULT): pure quadratic `float(self.w_soc) * float(soc - soc_target) ** 2` (db=0). The deadband form `float(self.w_soc) * k_inbounds * max(0.0, abs(soc - soc_target) - db) ** 2` is allowed for later refinement only.
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
1. Start from the baseline-intent weighted structure (this matches the ARROW baseline SAC reward):
   - base quadratic target-tracking term, pure form with NO deadband (db=0): `float(self.w_soc) * float(soc - soc_target) ** 2`
   - one continuous quadratic out-of-bounds strengthening term using `dist_low` / `dist_high`, also scaled by `self.w_soc`
   - do not use step multipliers or `np.where`-based weight switching (e.g. `w_soc * 10`) that create reward discontinuities at SOC boundaries
2. Out-of-bounds strengthening intent (mirror the baseline's harsh OOB penalty smoothly):
   - the baseline multiplies `w_soc` by ~10 outside `[soc_low, soc_high]`; reproduce this INTENT with a
     continuous term `self.w_soc * k_oob * (dist_low ** 2 + dist_high ** 2)` using a large `k_oob` (≈ 8–10)
   - this keeps SOC bounds quasi-hard (strong OOB pressure) WITHOUT the discontinuity, which is friendlier to SAC
   - the deadband form is NOT used in the initial reward; it is reserved as a later refinement-stage lever
3. Do not stack multiple high-SOC or asymmetry terms by default.
4. Keep out-of-bounds pressure much stronger than in-bounds pressure (this is the baseline intent).
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
0. Treat `objective_cost` as a real monetary cost in KRW (weights are market prices: `w_h2`≈10 KRW/g,
   `w_fc`≈4.275e6 KRW replacement, `w_batt`≈1.116e6 KRW replacement). Minimize total cost.
1. Keep objective decomposition explicit:
   - `objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost`
2. Keep the base weighted skeleton explicit in code:
   - `soc_cost` base (initial DEFAULT): pure quadratic `float(self.w_soc) * float(soc - soc_target) ** 2` (db=0); deadband form is a later-refinement lever only
   - `h2_cost = float(self.w_h2) * float(h2_fcs)`
   - `fcs_soh_cost = float(self.w_fc) * float(self.dSOH_FCS)`
   - `batt_soh_cost = float(self.w_batt) * float(self.dSOH_batt)`
3. Build `h2_cost` from pure `h2_fcs`, not from `h2_equal` and not as a separate top-level term.
4. Use degradation increments directly:
   - use `self.dSOH_FCS` and `self.dSOH_batt`
   - do not substitute `(SOH - 1)^2` proxies
5. Make hydrogen reduction the first objective.
6. Keep SOC charge-sustaining near `0.60`.
7. Keep `soc_cost` as a continuous quadratic structure (baseline-intent for the initial reward):
   - base quadratic tracking: pure form `float(self.w_soc) * float(soc - soc_target) ** 2` (db=0, tight tracking)
   - continuous out-of-bounds strengthening: add `self.w_soc * k_oob * (dist_low ** 2 + dist_high ** 2)` with `k_oob` ≈ 8–10
     (this reproduces the baseline's ~10× OOB penalty intent without the `np.where` discontinuity)
   - do not use step multipliers or `np.where`-based weight switching at SOC boundaries
   - the deadband form is reserved as a later refinement-stage lever; the initial reward uses db=0
8. Avoid reward hacking from signed battery credit:
   - charging above target should not become attractive just because `h2_batt` is negative
   - battery-equivalent terms are logged via `float(self.h2_batt)` / `float(self.h2_equal)` in `self.info` but must not be used as reward inputs
9. Avoid overly sharp or complicated penalties.
10. Keep the reward causal, stable, and easy to tune later.
11. Ensure code works in both training and smoke-test contexts.
12. Prefer simplicity over clever shaping.