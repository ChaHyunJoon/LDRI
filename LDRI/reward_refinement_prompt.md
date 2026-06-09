# Prompt for Iterative Reward Refinement (FCEV-EMS, SAC)

Use this prompt after each LDRI iteration to improve reward quality toward:
- lower hydrogen-related cost
- better preferred/dispreferred ordering under TPE replay
- SOC kept charge-sustaining (swing within bounds to enable battery load-leveling, not frozen near target)
- a simpler and more learnable reward

## 1) System Instruction (keep constant)

You are an expert in RL reward engineering and Python code editing.

Target code location:
- File: `FCHEV-EMS/LDRI/agentEMS_for_feedback.py`
- Class: `EMS`
- Method: `get_reward(self)`

PRIORITY HIERARCHY (non-negotiable):
  1. FUEL EFFICIENCY (h2_100km, g/km reduction) — PRIMARY OBJECTIVE
  2. SOC safety (OOB prevention) — CONSTRAINT to enable fuel efficiency, NOT a goal
  3. FCS durability — secondary; never at expense of fuel efficiency
  4. Training stability — tertiary

Primary objective for this project:
1. Minimize hydrogen consumption — PRIMARY GOAL:
   - prefer lower `h2_fcs` and lower `h2_cost`
   - keep `h2_cost` tied to pure `h2_fcs`, not `h2_equal`
   - do not create fake gains by charging the battery above target just to make `h2_batt` more negative
   - if hydrogen reduction conflicts with SOC staying near target, keep the solution charge-sustaining around `SOC_target`
   - target: `h2_cost` should account for 30–50% of total mean `objective_cost`; a fraction below 25% means H2 is underweighted
2. Battery SOC as an enabler of fuel efficiency — CONSTRAINT, NOT A GOAL:
   - SOC must stay inside `[soc_low, soc_high]` to prevent safety violations
   - SOC should SWING within `[soc_low, soc_high]` to enable battery load-leveling:
     - swing 0.05–0.25 per episode = HEALTHY — battery is buffering FCS load; do NOT tighten `soc_cost`
     - swing < 0.05 consistently = battery NOT buffering → FCS follows load → idling on low-speed cycles → H2 WASTE
     - if swing < 0.05 AND h2_100km is high, the root cause is over-tight `soc_cost`, not a hydrogen problem
   - do NOT tighten SOC tracking when battery buffering is healthy or swing is already small
   - do not over-penalize small transient in-bounds deviations if that breaks TPE ordering
3. Satisfy preferred/dispreferred ordering under TPE replay:
   - target condition: `preferred_avg > dispreferred_avg`
   - if a margin is provided, target condition becomes `preferred_avg > dispreferred_avg + margin`
   - if TPE says the preferred episode still gets lower reward, fixing that inversion is the first priority
4. Keep the additive objective fixed:
   - `objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost`
   - `reward = -objective_cost`
5. Preserve the weighted reward skeleton during refinement:
   - `soc_cost` anchored on `float(self.w_soc) * float(soc - soc_target) ** 2`
6. Tune degradation weights as a secondary fuel-efficiency lever:
   - if `mean_fcs_cost / mean_h2_cost > 0.15` consistently, FCS is idling or cycling excessively → increase `w_fc` by 10–25% to penalize this without adding new reward terms
   - `w_fc` tuning is secondary; always diagnose battery buffer utilization first
   - `batt_soh_cost` weight tuning is only warranted if battery degradation clearly dominates `objective_cost`

Hard constraints:
1. Edit only `EMS.get_reward(self)` in `FCHEV-EMS/LDRI/agentEMS_for_feedback.py`.
2. Keep exactly four top-level cost components:
   - `soc_cost`
   - `h2_cost`
   - `fcs_soh_cost`
   - `batt_soh_cost`
3. Keep pure fuel-cell hydrogen accounting inside `h2_cost` during refinement:
   - do not replace this with `h2_equal` during refinement
   - do not read `self.h2_batt` or `self.h2_equal` inside the reward computation logic
   - write `float(self.h2_batt)` and `float(self.h2_equal)` to `self.info` for logging only; do not use them as reward inputs
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
   - therefore read only `h2_fcs` for hydrogen logic
   - never read `self.h2_batt` or `self.h2_equal` inside reward computation; write `float(self.h2_batt)` and `float(self.h2_equal)` to `self.info` for logging only
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
11. Coefficient scale hard caps (absolute maximums — never exceed):
   - `k_oob` ≤ 8.0; if the current value already exceeds 8.0, reduce it before any other change
   - `k_inbounds` ≤ 5.0
   - effective `w_soc` multiplier ≤ 2.5× (relative to `self.w_soc`)
   - effective `w_h2` multiplier ≤ 4.0× (relative to `self.w_h2`)
   - per-iteration change rate: no coefficient may increase by more than 100% from its current value

SOC design rules:
1. Keep `soc_cost` simple and interpretable — at most 3 additive terms:
   - (1) base quadratic target-tracking term [REQUIRED]:
     - pure form: `w_soc * (soc - soc_target)^2`
     - deadband form: `w_soc * k_inbounds * max(0, |soc - soc_target| - db)^2`, where `db ∈ [0.01, 0.05]`
     - a wider deadband is preferred when battery buffering is suppressed (swing < 0.05), to free SOC to swing
   - (2) optional OOB quadratic strengthening: `w_soc * k_oob * (dist_low^2 + dist_high^2)` [REQUIRED when using deadband form]
   - (3) one optional bounded recovery shaping term [only when OOB rate > 5% in recent episodes]
   - if `soc_cost` already has more than 3 terms, remove the excess before adding anything
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
6. If both preferred and dispreferred episodes are already in-bounds and the preferred episode loses, inspect which term dominates before deciding what to soften or retune.
7. Battery buffer utilization diagnostic:
   - compute per-episode SOC swing: `soc_max - soc_min` from recent episodes
   - swing 0.05–0.25: HEALTHY — battery is load-leveling; do NOT tighten `soc_cost`
   - swing < 0.05 consistently AND h2_100km high → `soc_cost` is over-tight, battery not buffering:
     - CORRECT: reduce `w_soc` by 15–25%, or introduce/widen deadband `db ≈ 0.01–0.02`
     - FORBIDDEN: increasing `k_oob` or `k_inbounds` when swing is already small
   - swing > 0.30 AND OOB rate > 10% → SOC safety issue; increase `k_oob` conservatively

Refinement policy:
1. This is conservative refinement, not free-form redesign. Fuel efficiency (h2_100km, g/km) is always the primary objective; SOC is a constraint that enables it — do not allow `soc_cost` complexity to grow at the expense of the h2 optimization signal.
2. Diagnose battery buffer utilization before tuning any coefficient: if SOC swing < 0.05 and h2_100km is high, reduce `w_soc` or add deadband first — do NOT increase `w_h2` alone, as it will not address the root cause.
3. Focus on the reward terms most supported by the feedback; do not assume `soc_cost` is always the only lever.
4. Prefer simplifying existing shaping over adding new behavior terms.
5. If TPE reports negative `delta` or inverted ordering, solve that before optimizing any secondary metric.
6. If the preferred episode is better on hydrogen-related behavior but loses on reward, explain which term most likely caused the loss before changing the code.
7. Keep out-of-bounds pressure clearly stronger than in-bounds quadratic target tracking.
8. Avoid reward-scale blow-up and overly steep per-step penalties.
9. Preserve quadratic `soc_cost` structure because hydrogen / fuel-economy improvement is the first objective; do not trade it away for a linear SOC term unless numerical safety is at risk.
10. Prefer a small new local refinement over a no-op or historical revert.
11. A proposal is incomplete unless it explains why `preferred_avg - dispreferred_avg` should increase after the edit.

Output requirements:
1. `Analysis`
   - diagnose the current failure mode, including battery buffer utilization (SOC swing) and h2_cost fraction
   - identify which reward term likely caused the TPE inversion if one exists
   - explain why the proposed change should improve preferred/dispreferred ordering
2. `Updated Code`
   - provide full updated `get_reward(self)` only
3. `Expected Effects`
   - directional change for:
     - `preferred_avg - dispreferred_avg`
     - hydrogen-related cost (`h2_fcs` / `h2_cost`)
     - SOC tracking around `0.60`
     - battery buffer utilization (SOC swing)
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
1. reduce hydrogen-related cost (`h2_cost` should be 30–50% of mean `objective_cost`; below 25% means H2 is underweighted)
2. keep SOC charge-sustaining: SOC should swing within `[soc_low, soc_high]` for battery load-leveling, not stay frozen near target
3. repair TPE preferred/dispreferred ordering
4. keep the reward causal, simple, and learnable

Battery buffer utilization check (run this before choosing any fix):
- compute mean SOC swing: `soc_max - soc_min` from recent episodes in process feedback
- swing < 0.05 AND h2_100km high → `soc_cost` is over-tight; CORRECT ACTION: reduce `w_soc` or add deadband; do NOT increase `w_h2` alone
- swing healthy (0.05–0.25) AND h2 fraction < 25% → increase `w_h2` coefficient directly
- `mean_fcs_cost / mean_h2_cost` > 0.15 consistently → FCS idling is causing H2 waste; increase `w_fc` by 10–25%

If TPE reports negative `delta` or `preferred_avg < dispreferred_avg`:
- treat that as the immediate failure to repair
- explain which term made the preferred episode lose
- when both compared episodes are already in-bounds, inspect whether `soc_cost`, `h2_cost`, or degradation-related terms are driving the inversion before choosing a fix
- do not replace quadratic SOC deviation with linear / absolute-deviation penalties to repair TPE
- do not respond with progress-aware logic, terminal shaping, or a new complicated SOC sub-structure

If trajectory/preference feedback is unavailable:
- use episode-level feedback to infer whether the next change should retune `soc_cost`, hydrogen-related cost, degradation-related cost, or charge-sustaining SOC tracking
- check SOC swing in recent episodes: if swing < 0.05 consistently, `soc_cost` is over-tight regardless of h2 trend
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
   - if SOC swing < 0.05 AND h2_100km is high: fix `soc_cost` over-tightness first (reduce `w_soc` or add deadband); do NOT increase `w_h2` alone
4. keep SOC charge-sustaining: allow SOC to swing within bounds for battery load-leveling, not frozen near target
5. preserve a simple, causal, learnable reward

### Allowed Modification Scope
1. Tune `w_soc`, `k_inbounds`, `k_oob`, `db` (deadband), `w_h2`, `w_fc`, `w_batt` coefficients within scale caps (constraint #11).
2. Keep top-level objective decomposition unchanged and keep `h2_cost` based on pure `h2_fcs`.
3. Keep method signature and info contract unchanged.
4. Do not use progress-aware or terminal-aware shaping.
5. Do not broaden SOC structure when the issue can be solved by simplifying or retuning an existing quadratic term.

### Required Output Format
1. `Analysis`
2. `Updated Code`
3. `Expected Effects`
