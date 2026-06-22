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

PRIORITY STRUCTURE (non-negotiable):
  HARD CONSTRAINTS (above all objectives): numerical validity; SOC OOB safety (bounds always enforced)
  FOUR CO-OBJECTIVES (weighted; fuel consumption first-among-equals):
    - fuel_consumption (h2_100km, g/km) — wins DIRECT conflicts
    - soc_tracking, fc_degradation (w_fc·dSOH_FCS), battery_degradation (w_batt·dSOH_batt) — co-equal;
      each may be tuned (bounded) on its own merit when not directly opposing fuel_consumption
  TERTIARY: training stability
  'Fuel first' is a tie-break, NOT an absolute override.

Primary objective for this project:
1. Minimize hydrogen consumption — PRIMARY GOAL:
   - prefer lower `h2_fcs` and lower `h2_cost`
   - keep `h2_cost` tied to pure-hydrogen consumption `h2_fcs` (the fuel cell stack's actual H2 use)
   - `h2_batt` / `h2_equal` are still computed and logged for diagnostics, but are NOT priced into `h2_cost`
   - `soc_cost` anchors charge sustaining, but it does not prove charge sustaining by itself; high
     final/mean SOC with worse `h2_equal` is net charging and must be treated as a fuel regression
   - if hydrogen reduction conflicts with SOC staying near target, keep the solution charge-sustaining around `SOC_target`
   - target: `h2_cost` should account for 30–50% of total mean `objective_cost`; a fraction below 25% means H2 is underweighted
2. Battery SOC as an enabler of fuel efficiency — CONSTRAINT, NOT A GOAL:
   - SOC must stay inside `[soc_low, soc_high]` to prevent safety violations
   - SOC should SWING within `[soc_low, soc_high]` to enable battery load-leveling:
     - swing 0.05–0.25 per episode is healthy only when final/mean SOC stays near target and H2eq is not degrading
     - swing plus high final SOC is net charging, not healthy buffering
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
6. Tune degradation weights as co-objectives (each on its own merit, bounded):
   - if `mean_fcs_cost / mean_h2_cost > 0.15` consistently OR `mean_fcs_cost` trend rising → increase `w_fc` by 10–25% (no new reward terms)
   - if `mean_batt_cost / mean_h2_cost > 0.15` consistently OR `mean_batt_cost` trend rising → increase `w_batt` by 10–25% (no new reward terms)
   - degradation tuning is VALID without a direct h2 benefit; defer only on a direct conflict with fuel_consumption
   - per-iteration cap: w_fc and w_batt each ≤ 25% change; effective multiplier ≤ 3.0× the run_config baseline

Hard constraints:
1. Edit only `EMS.get_reward(self)` in `FCHEV-EMS/LDRI/agentEMS_for_feedback.py`.
2. Keep exactly four top-level cost components:
   - `soc_cost`
   - `h2_cost`
   - `fcs_soh_cost`
   - `batt_soh_cost`
3. Keep pure-hydrogen accounting inside `h2_cost` during refinement:
   - `h2_cost = w_h2 * h2_fcs` (pure hydrogen)
   - still compute `h2_batt = self.P_batt/1000 * self.eq_h2_batt_coef` and `h2_equal = h2_fcs + h2_batt` for diagnostics, but do NOT price them into `h2_cost`
   - use `h2_equal` diagnostically after training: if equivalent fuel worsens while final/mean SOC rises above target, the policy is net charging rather than charge sustaining
   - `h2_fcs` and `P_batt` are replay-compatible; also write `float(self.h2_batt)` and `float(self.h2_equal)` to `self.info`
   - do not add a battery-equivalent term into `h2_cost`; keep `h2_cost` a single pure-`h2_fcs` term
4. Keep `soc_cost`, `fcs_soh_cost`, and `batt_soh_cost` non-negative and finite.
   - keep the main SOC target-tracking term quadratic in SOC deviation (pure or deadband form)
   - out-of-bounds strengthening should remain quadratic by default:
     `w_soc * k_oob * (dist_low ** 2 + dist_high ** 2)`
   - do not switch to exponential OOB as a repair for high SOC, weak H2 pressure, or H2eq regression
   - do not replace the main target-tracking term with linear / absolute-deviation penalties during refinement
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
   - compute `h2_batt = self.P_batt/1000 * self.eq_h2_batt_coef` and `h2_equal = h2_fcs + h2_batt` from these replay-compatible inputs (for diagnostics only)
   - write `float(self.h2_batt)` and `float(self.h2_equal)` to `self.info`
   - do not rely on `step`, `episode_steps`, progress, travel, or terminal flags
7. Non-causal shaping is forbidden:
   - do not use episode progress, remaining route, remaining time, terminal bonuses, or late-episode gates
8. Return scalar `float`.
9. Code must run in both training and stub/smoke contexts.
10. Historical-reward regression is forbidden:
   - do not return the current reward unchanged when the prompt provides a concrete safe correction
   - do not restore any previously trained reward version verbatim
   - do not undo a prior iteration by reinstating an earlier exact coefficient/deadband setting
   - if no concrete safe correction is available, stop/retry review rather than inventing a patch
11. Coefficient scale hard caps (absolute maximums — never exceed):
   - `k_oob` ≤ 8.0; if the current value already exceeds 8.0, reduce it before any other change
   - `k_inbounds` ≤ 5.0
   - `soc_oob_exponent` (alpha in exponential OOB form) ≤ 3.0; assign as named constant `soc_oob_alpha = <value>`
   - effective `w_soc` multiplier ≤ 2.5× (relative to `self.w_soc`)
   - effective `w_h2` multiplier ≤ 4.0× (relative to `self.w_h2`)
   - `h2_cost` must stay LINEAR in `h2_fcs` (no power-law exponent)
   - per-iteration change rate: no coefficient may increase by more than 100% from its current value

SOC design rules:
1. Keep `soc_cost` non-dominating — `soc_cost / objective_cost` must stay below **0.85**:
   - if `soc_cost` already exceeds 85% of `objective_cost`, reduce `w_soc` or `k_inbounds` before any other change
   - (1) base target-tracking term [REQUIRED]:
     - pure quadratic: `w_soc * (soc - soc_target)^2`
     - deadband quadratic: `w_soc * k_inbounds * max(0, |soc - soc_target| - db)^2`, where `db ∈ [0.01, 0.05]`
     - a wider deadband is preferred when battery buffering is suppressed (swing < 0.05), to free SOC to swing
   - (2) optional OOB strengthening [REQUIRED when using deadband form]:
     - quadratic: `w_soc * k_oob * (dist_low^2 + dist_high^2)`
     - keep this quadratic unless a deterministic scale/safety reviewer explicitly requires otherwise
   - (3) one optional bounded recovery shaping term [only when OOB rate > 5% in recent episodes]
   - `soc_cost` is capped at three additive pieces: base target term, OOB quadratic, and at most one bounded recovery term
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
   - swing 0.05–0.25 is healthy only when final/mean SOC stays near target and H2eq is not degrading
   - swing plus high final SOC is net charging; do NOT treat it as healthy buffering
   - swing < 0.05 consistently AND h2_100km high → `soc_cost` is over-tight, battery not buffering:
     - CORRECT: reduce `w_soc` by 15–25%, or introduce/widen deadband `db ≈ 0.01–0.02`
     - FORBIDDEN: increasing `k_oob` or `k_inbounds` when swing is already small
   - high final/mean SOC (> target + 0.05) AND H2eq high/rising → net charging; rebalance h2_cost/SOC target pressure without increasing `k_oob`, `alpha`, or exponential OOB
   - swing > 0.30 AND OOB rate > 10% → SOC safety issue; increase quadratic `k_oob` conservatively

Refinement policy:
1. This is conservative refinement, not free-form redesign. Fuel efficiency is first-among-equals among charge-sustaining candidates; SOC OOB safety and high-SOC/H2eq regression cannot be sacrificed for apparent pure-H2 gains.
2. Diagnose battery buffer utilization before tuning any coefficient: if SOC swing < 0.05 and h2_100km is high, reduce `w_soc` or add deadband first — do NOT increase `w_h2` alone, as it will not address the root cause.
3. Focus on the reward terms most supported by the feedback; do not assume `soc_cost` is always the only lever.
4. Prefer simplifying existing shaping over adding new behavior terms.
5. If TPE reports negative `delta` or inverted ordering, solve that before optimizing any secondary metric.
6. If the preferred episode is better on hydrogen-related behavior but loses on reward, explain which term most likely caused the loss before changing the code.
7. Keep out-of-bounds pressure clearly stronger than in-bounds quadratic target tracking.
8. Avoid reward-scale blow-up and overly steep per-step penalties.
9. Preserve quadratic `soc_cost` structure because hydrogen / fuel-economy improvement is the first objective; do not trade it away for a linear SOC term unless numerical safety is at risk.
10. Prefer a small new local refinement over a historical revert when a concrete safe correction is specified; otherwise stop/retry rather than inventing a no-op-breaking edit.
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
- swing 0.05–0.25 is healthy only if final/mean SOC stays near target and H2eq is not degrading
- high final/mean SOC (> target + 0.05) AND H2eq high/rising → net charging; rebalance charge sustaining without increasing `k_oob`, `alpha`, or exponential OOB
- swing healthy AND h2 fraction < 25% → increase `w_h2` coefficient directly
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
- returning the current reward unchanged is invalid only when a concrete safe correction is specified
- returning any previously trained reward variant is invalid
- if a previous iteration already used a coefficient/deadband setting, do not restore that exact setting as the new "improvement"
- when a recent edit looks wrong and a safe correction is clear, fix it with a new local change rather than rolling back to an older trained reward
- if no concrete safe correction is available, stop/retry review rather than inventing a patch

### Mandatory Decision Order
If signals conflict, choose in this order:
1. numerical safety and implementation validity
2. repair TPE preferred/dispreferred ordering
3. reduce hydrogen-related cost without causing obvious SOC drift
   - if SOC swing < 0.05 AND h2_100km is high: fix `soc_cost` over-tightness first (reduce `w_soc` or add deadband); do NOT increase `w_h2` alone
   - if final/mean SOC is high and H2eq is high/rising: treat as equivalent-fuel regression; do NOT increase `k_oob`, `alpha`, or exponential OOB
4. keep SOC charge-sustaining: allow SOC to swing within bounds for battery load-leveling, not frozen near target
5. preserve a simple, causal, learnable reward

### Allowed Modification Scope
1. Tune only coefficients supported by the feedback: `w_soc`, `k_inbounds`, `k_oob`, `db` (deadband), `w_h2`, `w_fc`, `w_batt` within scale caps (constraint #11). Do not switch to or increase exponential OOB unless deterministic safety review explicitly requires it.
2. Keep top-level objective decomposition unchanged.
   `h2_cost` must be the **fixed-structure pure-hydrogen term**, linear in `h2_fcs`:
   - `effective_w_h2 * h2_fcs` (`h2_batt`/`h2_equal` stay diagnostic-only, not priced into `h2_cost`)
   - monotone non-decreasing in `h2_fcs`; keep LINEAR (no power-law on `h2_fcs`)
   — do NOT add EMA baselines, efficiency bonuses, activity gates, or state-dependent multipliers.
3. Keep method signature and info contract unchanged.
4. Do not use progress-aware or terminal-aware shaping.
5. Do not broaden SOC structure when the issue can be solved by simplifying or retuning an existing term.

### Required Output Format
1. `Analysis`
2. `Updated Code`
3. `Expected Effects`
