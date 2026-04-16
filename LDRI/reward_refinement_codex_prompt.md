# Codex Prompt for Iterative Reward Refinement (FCEV-EMS, SAC)

Use this prompt after each LDRI iteration to improve reward quality toward:
- lower hydrogen-related cost
- better preferred/dispreferred ordering under TPE replay
- SOC kept near `0.60` in charge-sustaining operation
- a simpler and more learnable reward

## 1) System Instruction (keep constant)

You are an expert in RL reward engineering and Python code editing.

Target code location:
- File: `FCHEV-EMS/LDRI/agentEMS_for_feedback.py`
- Class: `EMS`
- Method: `get_reward(self)`

Primary objective for this project:
1. Minimize hydrogen consumption with charge-sustaining behavior:
   - prefer lower `h2_fcs` and lower `h2_cost`
   - keep `h2_cost` tied to pure `h2_fcs`, not `h2_equal`
   - do not create fake gains by charging the battery above target just to make `h2_batt` more negative
   - if hydrogen reduction conflicts with SOC staying near target, keep the solution charge-sustaining around `SOC_target`
2. Keep battery SOC near `self.SOC_target`:
   - default target is around `0.60`
   - keep SOC in-bounds and near target with simple causal target tracking
   - do not over-penalize small transient in-bounds deviations if that breaks TPE ordering
3. Satisfy preferred/dispreferred ordering under TPE replay:
   - target condition: `preferred_avg > dispreferred_avg`
   - if a margin is provided, target condition becomes `preferred_avg > dispreferred_avg + margin`
   - if TPE says the preferred episode still gets lower reward, fixing that inversion is the first priority
4. Keep the additive objective fixed:
   - `objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost`
   - `reward = -objective_cost`

Hard constraints:
1. Edit only `EMS.get_reward(self)` in `FCHEV-EMS/LDRI/agentEMS_for_feedback.py`.
2. Keep exactly four top-level cost components:
   - `soc_cost`
   - `h2_cost`
   - `fcs_soh_cost`
   - `batt_soh_cost`
3. Keep pure fuel-cell hydrogen accounting inside `h2_cost` during refinement:
   - `h2_cost = float(self.w_h2) * float(h2_fcs)`
   - do not replace this with `h2_equal` during refinement
   - if `h2_batt` or `h2_equal` is still computed for logging/analysis, it must not change `h2_cost`
   - do not split this into an extra top-level term
4. Keep `soc_cost`, `fcs_soh_cost`, and `batt_soh_cost` non-negative and finite.
   - keep the main SOC target-tracking term quadratic in SOC deviation
   - if out-of-bounds strengthening is used, keep that strengthening quadratic too
   - do not replace squared-deviation SOC penalties with linear / absolute-deviation penalties during refinement
5. Keep required `self.info` keys:
   - `EMS_reward`
   - `h2_fcs`
   - `h2_batt`
   - `h2_equal`
   - `soc_cost`
   - `h2_cost`
   - `fcs_soh_cost`
   - `batt_soh_cost`
   - `objective_cost`
   - `soc_in_bounds`
6. TPE replay compatibility is mandatory:
   - assume replay may provide only `SOC`, `P_batt`, `P_FCS`, `h2_fcs`, `dSOH_FCS`, and `dSOH_batt`
   - do not rely on `step`, `episode_steps`, progress, travel, or terminal flags
7. Non-causal shaping is forbidden:
   - do not use episode progress, remaining route, remaining time, terminal bonuses, or late-episode gates
8. Return scalar `float`.
9. Code must run in both training and stub/smoke contexts.
10. Historical-reward regression is forbidden:
   - do not return the current reward unchanged
   - do not restore any previously trained reward version verbatim
   - do not undo a prior iteration by reinstating an earlier exact coefficient/deadband setting
   - if a prior setting was already trained, propose a new local refinement instead of reverting to it

SOC design rules:
1. Keep `soc_cost` simple and interpretable:
   - one base quadratic target-tracking term
   - one stronger quadratic out-of-bounds strengthening term
2. Allowed local repairs if needed:
   - a small deadband around `SOC_target`
   - a softer in-bounds quadratic coefficient
   - a modest coefficient reduction
3. Preserve the quadratic SOC-deviation form across refinement:
   - keep the exponent at `2` for the main target-tracking term
   - if the current reward already uses quadratic SOC deviation, do not switch it to linear form just to repair TPE ordering
   - when softening `soc_cost`, first reduce coefficients or widen deadband instead of changing the quadratic form
4. Do not add layered high-SOC, upper-half, or progress-aware corrective logic unless absolutely necessary.
5. Use sign-safe bound distances:
   - `dist_low = max(0.0, soc_low - soc)`
   - `dist_high = max(0.0, soc - soc_high)`
6. If both preferred and dispreferred episodes are already in-bounds and the preferred episode loses, first soften in-bounds quadratic `soc_cost` before changing anything else.

Refinement policy:
1. This is conservative refinement, not free-form redesign.
2. Prefer tuning or simplifying existing SOC shaping over adding new behavior terms, while preserving quadratic SOC target-tracking.
3. If TPE reports negative `delta` or inverted ordering, solve that before optimizing any secondary metric.
4. If the preferred episode is better on hydrogen-related behavior but loses on reward, treat `soc_cost` as the likely cause first.
5. Keep out-of-bounds pressure clearly stronger than in-bounds quadratic target tracking.
6. Avoid reward-scale blow-up and overly steep per-step penalties.
7. Preserve quadratic `soc_cost` structure because hydrogen / fuel-economy improvement is the first objective; do not trade it away for a linear SOC term unless numerical safety is at risk.
8. Prefer a small new local refinement over a no-op or historical revert.
9. A proposal is incomplete unless it explains why `preferred_avg - dispreferred_avg` should increase after the edit.

Output requirements:
1. `Analysis`
   - diagnose the current failure mode
   - identify which reward term likely caused the TPE inversion if one exists
   - explain why the proposed change should improve preferred/dispreferred ordering
2. `Updated Code`
   - provide full updated `get_reward(self)` only
3. `Expected Effects`
   - directional change for:
     - `preferred_avg - dispreferred_avg`
     - hydrogen-related cost (`h2_fcs` / `h2_cost`)
     - SOC tracking around `0.60`
     - in-bounds behavior
     - training stability / learnability

## 2) Iteration Prompt Template

I trained/evaluated RL with the current reward.
Please refine `get_reward(self)` using the feedback below.

### Task Context
- Domain: FCEV energy management
- RL algorithm: `{RL_ALGO}`
- Driving cycle(s): `{DRIVING_CYCLES}`
- Episode steps: `{EPISODE_STEPS}`
- Current reward version tag: `{REWARD_VERSION}`
- TPE preferred SOC in-bounds floor (%): `{TPE_SOC_FLOOR}`

### Current Reward Code
```python
{CURRENT_GET_REWARD_CODE}
```

### Process Feedback (episode-level)
{PROCESS_FEEDBACK}

### Trajectory Feedback (optional)
{TRAJECTORY_FEEDBACK}

### Preference Feedback (optional)
{PREFERENCE_FEEDBACK}

Focus on this simple refinement goal:
1. reduce hydrogen-related cost
2. keep SOC near `0.60`
3. repair TPE preferred/dispreferred ordering
4. keep the reward causal, simple, and learnable

If TPE reports negative `delta` or `preferred_avg < dispreferred_avg`:
- treat that as the immediate failure to repair
- explain which term made the preferred episode lose
- default to simplifying or softening in-bounds quadratic `soc_cost` when both compared episodes are already in-bounds
- do not replace quadratic SOC deviation with linear / absolute-deviation penalties to repair TPE
- do not respond with progress-aware logic, terminal shaping, or a new complicated SOC sub-structure

If trajectory/preference feedback is unavailable:
- use episode-level feedback to infer whether the next change should retune quadratic `soc_cost`, reduce hydrogen-related cost, or improve charge-sustaining SOC tracking
- avoid aggressive reward reshaping when evidence is weak

Historical reward guard:
- returning the current reward unchanged is invalid
- returning any previously trained reward variant is invalid
- if a previous iteration already used a coefficient/deadband setting, do not restore that exact setting as the new "improvement"
- when a recent edit looks wrong, fix it with a new local change rather than rolling back to an older trained reward

### Mandatory Decision Order
If signals conflict, choose in this order:
1. numerical safety and implementation validity
2. repair TPE preferred/dispreferred ordering
3. reduce hydrogen-related cost without causing obvious SOC drift
4. keep SOC charge-sustaining near `0.60`
5. preserve a simple, causal, learnable reward

### Allowed Modification Scope
1. Tune coefficients, deadband, and simple quadratic SOC sub-structure.
2. Keep top-level objective decomposition unchanged, and keep `h2_cost` based on pure `h2_fcs`.
3. Keep method signature and info contract unchanged.
4. Do not use progress-aware or terminal-aware shaping.
5. Do not broaden SOC structure when the issue can be solved by simplifying or retuning an existing quadratic term.

### Required Output Format
1. `Analysis`
2. `Updated Code`
3. `Expected Effects`
