import numpy as np
from FCHEV_SOH import FCHEV_SOH
from Battery import CellModel1


class EMS:
    """ EMS with SOH """
    def __init__(
        self,
        w_soc,
        soc0,
        SOC_MODE,
        abs_spd_MAX,
        abs_acc_MAX,
        w_h2=10.0,        # 전국 수소충전소 평균 단가: 10원/g
        w_fc=4.275e6,     # DoE 기준 2025년 상용차 PEMFC 양산단가: (30$)4.5e4원/kW -> 연료전지 전체 교체비용을 계수로 잡으면 4.275e6
        w_batt=1.116e6,     # BloombergNEF 기준 2025년 Li이온 배터리팩 평균 가격: 1.5e5원/kWh -> 배터리 전체 교체비용을 계수로 잡으면 1.116e6
        eq_h2_batt_coef=0.0164,
        soc_target=0.6,
        soc_bounds=(0.4, 0.8),
    ):
        self.time_step = 1.0
        self.w_soc = w_soc
        self.w_h2 = w_h2
        self.w_fc = w_fc
        self.w_batt = w_batt
        self.eq_h2_batt_coef = max(0.0, float(eq_h2_batt_coef))
        soc_low, soc_high = soc_bounds
        if soc_low > soc_high:
            soc_low, soc_high = soc_high, soc_low
        self.soc_bounds = (float(soc_low), float(soc_high))
        self.soc_weight_multiplier = 1.0
        self.done = False
        self.info = {}
        self.FCHEV = FCHEV_SOH()
        self.Battery = CellModel1()
        self.obs_num = 7  # soc, soh-batt, soh-fcs, P_FCS(t-1), P_batt, spd, acc
        self.action_num = 1  # P_FCS (t)
        # motor, unit in W
        self.P_mot_max = self.FCHEV.motor_max_power  # 200,000 W
        self.P_mot = 0
        # FCS, unit in kW
        self.h2_fcs = 0
        self.h2_batt = 0
        self.h2_equal = 0
        self.P_FCS = 0
        self.P_FCS_max = self.FCHEV.P_FC_max        # kW
        self.dSOH_FCS = 0
        self.SOH_FCS = 1.0
        # battery unit in W
        self.SOC_init = soc0        # 0.6
        if soc_target is not None:
            self.SOC_target = float(soc_target)
        elif SOC_MODE == 'CD':        # charge-depletion
            self.SOC_target = self.SOC_init - 0.2
        else:                         # charge-sustaining
            self.SOC_target = self.SOC_init
        self.SOC = self.SOC_init
        self.OCV_initial = self.Battery.ocv_func(self.SOC_init)
        self.SOH_batt = 1.0
        self.Tep_a = 20
        self.P_batt = 0     # W
        self.P_batt_max = self.Battery.batt_maxpower    # in W
        self.SOC_delta = 0
        self.dSOH_batt = 0
        self.I_batt = 0
        # paras_list = [SOC, SOH, Voc]
        self.paras_list = [self.SOC, self.SOH_batt, self.OCV_initial]
        self.travel = 0
        self.car_spd = 0
        self.car_acc = 0
        self.abs_spd_MAX = abs_spd_MAX
        self.abs_acc_MAX = abs_acc_MAX
    
    def reset_obs(self):
        self.SOC = self.SOC_init
        self.SOH_batt = 1.0
        self.SOH_FCS = 1.0
        self.dSOH_FCS = 0
        self.Tep_a = 20
        self.P_mot = 0
        self.P_FCS = 0
        self.P_batt = 0
        self.h2_fcs = 0
        self.h2_batt = 0
        self.h2_equal = 0
        self.FCHEV.P_FC_old = 0.0
        self.paras_list = [self.SOC, self.SOH_batt, self.OCV_initial]
        self.done = False
        self.info = {}
        self.travel = 0
        self.car_spd = 0
        self.car_acc = 0
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        # soc, soh-batt, soh-fcs, P_FCS, P_batt, spd, acc
        obs[0] = self.SOC
        obs[1] = self.SOH_batt
        obs[2] = self.SOH_FCS
        obs[3] = self.P_FCS / self.P_FCS_max        # in kW
        obs[4] = self.P_batt / self.P_batt_max   # in W
        obs[5] = self.car_spd / self.abs_spd_MAX
        obs[6] = self.car_acc / self.abs_acc_MAX
        return obs

    def execute(self, action, car_spd, car_acc):
        self.car_spd = car_spd
        self.car_acc = car_acc
        # Action(-1 ~ 1)을 0 ~ 1 비율로 선형 변환
        action_mapped = (action[0] + 1.0) / 2.0
        # 클리핑(안전장치): 범위를 벗어나는 혹시 모를 상황 방지
        action_mapped = np.clip(action_mapped, 0.0, 1.0) 
        self.P_FCS = action_mapped * self.FCHEV.P_FC_max     # kW

        T_axle, W_axle, P_axle = self.FCHEV.T_W_axle(self.car_spd, self.car_acc)
        T_mot, W_mot, mot_eff, self.P_mot = self.FCHEV.run_motor(T_axle, W_axle, P_axle)  # W
        P_dcdc, self.h2_fcs, info_fcs = self.FCHEV.run_fuel_cell(self.P_FCS)      # kW
        self.dSOH_FCS, info_fcs_soh = self.FCHEV.run_FC_SOH(self.P_FCS)
        self.SOH_FCS -= self.dSOH_FCS
        self.P_batt = self.P_mot - P_dcdc*1000        # W
        # update power battery
        self.paras_list, self.dSOH_batt, self.I_batt, self.done, info_batt = \
            self.Battery.run_battery(self.P_batt, self.paras_list)
        self.P_batt = float(info_batt['P_batt_req']) * 1000.0
        self.SOC = self.paras_list[0]
        self.SOH_batt = self.paras_list[1]
        self.Tep_a = 20.0
        self.h2_batt = float(self.P_batt) / 1000.0 * self.eq_h2_batt_coef
        self.h2_equal = float(self.h2_fcs) + self.h2_batt

        self.travel += self.car_spd*self.time_step
        self.info = {}
        self.info.update({'T_axle': T_axle, 'W_axle': W_axle, 'P_axle': P_axle/1000,
                          'T_mot': T_mot, 'W_mot': W_mot, 'mot_eff': mot_eff,
                          'P_mot': self.P_mot/1000, 'FCS_SOH': self.SOH_FCS,
                          'travel': self.travel})
        self.info.update(info_fcs)
        self.info.update(info_batt)
        self.info.update(info_fcs_soh)
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        # soc, soh-batt, soh-fcs, P_FCS, P_batt, spd, acc -> normalized within [0,1] range
        obs[0] = self.SOC
        obs[1] = self.SOH_batt
        obs[2] = self.SOH_FCS
        obs[3] = self.P_FCS / self.P_FCS_max        # in kW
        obs[4] = self.P_batt / self.P_batt_max   # in W
        obs[5] = self.car_spd / self.abs_spd_MAX
        obs[6] = self.car_acc / self.abs_acc_MAX
        return obs
    
    def get_info(self):
        return self.info

    def get_done(self):
        return self.done
    def get_reward(self):
        """
        Reward function for FCEV-EMS (SAC, iter_005).

        Objective (minimized):
            objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost

        Design principles:
        - h2_cost: pure FC hydrogen consumption only (h2_fcs), no battery credit/debit.
        - soc_cost: deadband quadratic base (free swing within ±db of target) +
                    continuous quadratic out-of-bounds strengthening.
                    Deadband allows healthy battery buffering; k_oob guards [0.4, 0.8].
        - fcs_soh_cost / batt_soh_cost: direct per-step degradation increments.
        - h2_batt and h2_equal are logged for analysis only; never used as reward inputs.
        - Fully causal: no step index, no episode progress, no terminal logic.

        Changes from iter_004:
        - k_oob: 4.0 -> 7.5 (CRITICAL FIX — iter_004 oob_rate=93.8% / soc_spiral failure.
          OOB guard was too weak; SOC left [0.4, 0.8] almost every step. Restoring strong
          OOB protection. 7.5 is within 8.0 hard cap and within 2x per-iter limit from 4.0.)
        - k_inbounds: 0.3 -> 0.15 (further soften in-bounds base to keep soc_cost share low
          when SOC is inside bounds; OOB guard now carries the safety burden.)
        - h2_scale: 2.0 -> 3.5 (increase h2 weight to make h2_cost a stronger episode-level
          differentiator — TAC Kendall-tau=0.005 indicates reward does not rank episodes by
          fuel economy. Larger h2_cost share improves rank correlation. 3.5 < 4.0 hard cap;
          change is +75% from 2.0, within 100% per-iter limit.)
        - db: 0.015 -> 0.010 (tighten deadband slightly — with strong k_oob restored, a
          narrower free zone keeps SOC closer to target while still allowing buffering swing.)

        Rationale:
          iter_004 suffered a soc_spiral (oob_rate=93.8%) because k_oob was reduced too
          aggressively (4.5->4.0) while k_inbounds was also cut (0.5->0.3). The combined
          weakening of both SOC terms left no effective restoring force when SOC drifted OOB.
          Fix: restore k_oob=7.5 (strong OOB guard), reduce k_inbounds=0.15 (minimal in-bounds
          pressure), and raise h2_scale=3.5 so h2_cost dominates the objective when SOC is
          in bounds. This directly addresses TAC failure: episodes with lower h2_100km will
          accumulate less h2_cost and receive higher reward, improving Kendall-tau rank
          correlation. The db narrowing to 0.010 pairs with the stronger k_oob to keep SOC
          well-contained without over-penalizing healthy swing.
        """

        # ------------------------------------------------------------------
        # 1) Read core states
        # ------------------------------------------------------------------
        soc        = float(self.SOC)
        soc_low    = float(self.soc_bounds[0])   # 0.40
        soc_high   = float(self.soc_bounds[1])   # 0.80
        soc_target = float(self.SOC_target)      # 0.60

        # ------------------------------------------------------------------
        # 2) Hydrogen term (pure FC consumption, no battery equivalent)
        # ------------------------------------------------------------------
        h2_fcs = float(self.h2_fcs)             # kg/step, non-negative

        # ------------------------------------------------------------------
        # 3) SOC cost — deadband quadratic structure
        #
        #    Term 1 (base): deadband quadratic — zero penalty inside ±db of target,
        #                   allowing the battery to buffer FCS load freely.
        #        soc_cost_base = w_soc * k_inbounds * max(0, |soc - soc_target| - db)^2
        #
        #    Term 2 (OOB):  out-of-bounds strengthening — activates only when
        #                   soc < soc_low or soc > soc_high.
        #        soc_cost_oob = w_soc * k_oob * (dist_low^2 + dist_high^2)
        #
        #    db = 0.010: ±1.0% SOC free zone near target (tightened from 0.015)
        #    k_inbounds = 0.15: minimal in-bounds pressure; OOB guard carries safety burden.
        #    k_oob = 7.5: strong OOB guard — critical fix for iter_004 soc_spiral.
        #                 Within 8.0 hard cap; +87.5% from 4.0 (within 100% per-iter limit).
        # ------------------------------------------------------------------
        db         = 0.010   # deadband half-width: ±1.0% SOC free zone
        k_inbounds = 0.15    # in-bounds base multiplier (softened from 0.3)
        k_oob      = 7.5     # OOB strengthening multiplier (restored strong; was 4.0)

        dist_low  = max(0.0, soc_low  - soc)          # > 0 only when soc < soc_low
        dist_high = max(0.0, soc      - soc_high)      # > 0 only when soc > soc_high
        dev       = max(0.0, abs(soc - soc_target) - db)  # deviation beyond deadband

        soc_cost = (
            float(self.w_soc) * k_inbounds * dev ** 2
            + float(self.w_soc) * k_oob * (dist_low ** 2 + dist_high ** 2)
        )

        # ------------------------------------------------------------------
        # 4) Hydrogen cost — pure FC hydrogen, no battery credit/debit
        #    h2_scale: 2.0 -> 3.5 (increase to make h2_cost dominant episode differentiator)
        #    Effective w_h2 = 10.0 * 3.5 = 35.0
        #    Within 4.0x env cap (max 40.0); +75% from 2.0, within 100% per-iter limit.
        # ------------------------------------------------------------------
        h2_scale = 3.5
        h2_cost  = float(self.w_h2) * h2_scale * float(h2_fcs)

        # ------------------------------------------------------------------
        # 5) Degradation costs — direct per-step increments (non-negative)
        # ------------------------------------------------------------------
        fcs_soh_cost  = float(self.w_fc)   * float(self.dSOH_FCS)
        batt_soh_cost = float(self.w_batt) * float(self.dSOH_batt)

        # ------------------------------------------------------------------
        # 6) Aggregate objective and reward
        # ------------------------------------------------------------------
        objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost
        reward         = float(-objective_cost)

        # ------------------------------------------------------------------
        # 7) Logging — h2_batt and h2_equal are written here for analysis
        #    only; they are NOT used anywhere in the reward computation above.
        # ------------------------------------------------------------------
        self.info.update({
            "EMS_reward":    reward,
            "h2_fcs":        float(h2_fcs),
            "h2_batt":       float(self.h2_batt),    # log only
            "h2_equal":      float(self.h2_equal),   # log only
            "soc_cost":      float(soc_cost),
            "h2_cost":       float(h2_cost),
            "fcs_soh_cost":  float(fcs_soh_cost),
            "batt_soh_cost": float(batt_soh_cost),
            "objective_cost": float(objective_cost),
            "soc_in_bounds": int(soc_low < soc < soc_high),
        })

        return reward
    