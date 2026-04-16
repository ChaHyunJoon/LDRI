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
        soc = float(self.SOC)
        soc_low, soc_high = self.soc_bounds
        soc_target = float(self.SOC_target)

        # Hydrogen accounting
        h2_fcs = max(0.0, float(self.h2_fcs))
        h2_batt = float(self.eq_h2_batt_coef) * float(self.P_batt) / 1000.0 / 3600.0
        h2_equal = float(h2_fcs + h2_batt)

        # SOC regulation:
        # - simple quadratic target tracking in-bounds
        # - small deadband to avoid over-penalizing tiny in-bounds fluctuations
        # - stronger quadratic strengthening when out of bounds
        deadband = 0.008
        soc_err = abs(soc - soc_target)
        soc_err_eff = max(0.0, soc_err - deadband)

        dist_low = max(0.0, float(soc_low) - soc)
        dist_high = max(0.0, soc - float(soc_high))

        in_bounds_track = float(self.w_soc) * 0.85 * (soc_err_eff ** 2)
        out_of_bounds_strength = float(self.w_soc) * 12.0 * ((dist_low ** 2) + (dist_high ** 2))
        soc_cost = max(0.0, in_bounds_track + out_of_bounds_strength)

        # Pure fuel-cell hydrogen cost only
        h2_cost = float(self.w_h2) * float(h2_fcs)

        # Degradation costs
        fcs_soh_cost = max(0.0, float(self.w_fc) * max(0.0, float(self.dSOH_FCS)))
        batt_soh_cost = max(0.0, float(self.w_batt) * max(0.0, float(self.dSOH_batt)))

        objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost
        reward = float(-objective_cost)

        self.info.update({
            "EMS_reward": reward,
            "h2_fcs": float(h2_fcs),
            "h2_batt": float(h2_batt),
            "h2_equal": float(h2_equal),
            "soc_cost": float(soc_cost),
            "h2_cost": float(h2_cost),
            "fcs_soh_cost": float(fcs_soh_cost),
            "batt_soh_cost": float(batt_soh_cost),
            "objective_cost": float(objective_cost),
            "soc_in_bounds": int(soc_low < soc < soc_high),
        })
        return reward
    