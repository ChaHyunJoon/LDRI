import numpy as np
from FCHEV_SOH import FCHEV_SOH
from Battery import CellModel1


class DP_EMS_agent:
    def __init__(
        self,
        w_soc=100,
        soc0=0.6,
        SOC_MODE="CS",
        soc_ref=1.0,
        w_h2=10.0,        # 전국 수소충전소 평균 단가: 10원/g
        w_fc=4.275e6,     # DoE 기준 2025년 상용차 PEMFC 양산단가: (30$)4.5e4원/kW -> 연료전지 전체 교체비용을 계수로 잡으면 4.275e6
        w_batt=1.116e6,     # BloombergNEF 기준 2025년 Li이온 배터리팩 평균 가격: 1.5e5원/kWh -> 배터리 전체 교체비용을 계수로 잡으면 1.116e6
        eq_h2_batt_coef=0.0164,
        soc_target=0.6,
        soc_bounds=(0.4, 0.8)
    ):
        self.FCHEV = FCHEV_SOH()
        self.Battery = CellModel1()
        self.info = {}
        self.w_soc = w_soc
        self.w_h2 = w_h2
        self.w_fc = w_fc
        self.w_batt = w_batt
        self.eq_h2_batt_coef = max(0.0, float(eq_h2_batt_coef))
        soc_low, soc_high = soc_bounds
        if soc_low > soc_high:
            soc_low, soc_high = soc_high, soc_low
        self.soc_bounds = (float(soc_low), float(soc_high))
        self.soc_ref = soc_ref
        # EMS
        if soc_target is not None:
            self.SOC_target = soc_target
        elif SOC_MODE == "CD":
            self.SOC_target = soc0 - 0.2
        else:
            self.SOC_target = soc0
        self.SOC = self.SOC_target
        self.h2_fcs = 0
        self.h2_batt = 0
        self.h2_equal = 0
        self.dSOH_FCS = 0
        self.P_batt = 0
        self.dSOH_batt = 0
        self._bw_cache = None

    def _compute_fc_soh_batch(self, P_FCS, P_fc_old):
        P_fc_low = self.FCHEV.P_FC_low
        P_fc_high = self.FCHEV.P_FC_high
        P_fc_off = self.FCHEV.P_FC_off
        simtime = self.FCHEV.simtime
        on_old = bool(P_fc_old >= P_fc_off)
        on_new = P_FCS >= P_fc_off
        d_s_s = np.where((not on_old) & on_new, simtime * 1.96, 0.0)
        d_low = np.where(on_new & (P_FCS < P_fc_low), simtime * 1.26, 0.0)
        d_high = np.where(P_FCS >= P_fc_high, simtime * 1.47, 0.0)
        d_l_c = 5.93 * self.FCHEV.load_change_cycle_count(P_FCS, P_fc_old)
        De_i = (d_s_s + d_low + d_high) * 1e-3 / 3600 + d_l_c * 1e-5
        return De_i / 100.0

    def prepare_backward_cache(self, actions, speed_list, acc_list, states):
        actions_arr = np.asarray(actions, dtype=np.float64).reshape(-1)
        speed_arr = np.asarray(speed_list, dtype=np.float64).reshape(-1)
        acc_arr = np.asarray(acc_list, dtype=np.float64).reshape(-1)
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1)

        actions_mapped = np.clip((actions_arr + 1.0) / 2.0, 0.0, 1.0)
        P_FCS = actions_mapped * self.FCHEV.P_FC_max
        P_dcdc, h2_fcs, _ = self.FCHEV.run_fuel_cell(P_FCS)
        P_dcdc = np.asarray(P_dcdc, dtype=np.float64).reshape(-1)
        h2_fcs = np.asarray(h2_fcs, dtype=np.float64).reshape(-1)
        dsoh_fcs = self._compute_fc_soh_batch(P_FCS, float(self.FCHEV.P_FC_old)).reshape(-1)

        P_mot_t = np.zeros_like(speed_arr, dtype=np.float64)
        for idx, (spd, acc) in enumerate(zip(speed_arr, acc_arr)):
            T_axle, W_axle, P_axle = self.FCHEV.T_W_axle(float(spd), float(acc))
            _, _, _, P_mot = self.FCHEV.run_motor(T_axle, W_axle, P_axle)
            P_mot_t[idx] = float(P_mot)
        P_batt_ta = np.clip(
            P_mot_t[:, None] - P_dcdc[None, :] * 1000.0,
            -self.Battery.batt_maxpower,
            self.Battery.batt_maxpower,
        )
        h2_batt_ta = np.maximum(P_batt_ta, 0.0) / 1000.0 * self.eq_h2_batt_coef
        h2_equal_ta = h2_fcs[None, :] + h2_batt_ta

        ocv_states = np.asarray(self.Battery.ocv_func(states_arr), dtype=np.float64)
        r0_states = np.asarray(self.Battery.r0_func(states_arr), dtype=np.float64)
        low_mask = states_arr <= self.Battery._loss_soc_split
        c1_states = np.where(low_mask, self.Battery._loss_c1_low, self.Battery._loss_c1_high)
        c2_states = np.where(low_mask, self.Battery._loss_c2_low, self.Battery._loss_c2_high)

        self._bw_cache = {
            "actions": actions_arr,
            "states": states_arr,
            "h2_fcs": h2_fcs,
            "dsoh_fcs": dsoh_fcs,
            "p_batt_ta": P_batt_ta,
            "h2_batt_ta": h2_batt_ta,
            "h2_equal_ta": h2_equal_ta,
            "ocv_states": ocv_states,
            "r0_states": r0_states,
            "c1_states": c1_states,
            "c2_states": c2_states,
        }
        return self._bw_cache

    def batch_transition_reward(self, time_step, state_ids):
        if self._bw_cache is None:
            raise RuntimeError("backward cache is not prepared; call prepare_backward_cache() first")

        ids = np.asarray(state_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            empty = np.empty((0, self._bw_cache["actions"].shape[0]), dtype=np.float64)
            return empty, empty

        cache = self._bw_cache
        p_batt = cache["p_batt_ta"][int(time_step)][None, :]

        soc_arr = cache["states"][ids][:, None]
        voc = cache["ocv_states"][ids][:, None]
        r0 = cache["r0_states"][ids][:, None]
        delta = voc ** 2 - 4.0 * r0 * p_batt
        sqrt_delta = np.sqrt(np.maximum(delta, 0.0))
        I_batt = np.where(delta < 0.0, voc / (2.0 * r0), (voc - sqrt_delta) / (2.0 * r0))

        soc_deriv = self.Battery.timestep * (I_batt / 3600.0 / self.Battery.Cn)
        soc_new = np.clip(soc_arr - soc_deriv, self.Battery.soc_min, self.Battery.soc_max)

        current_step_Ah = (np.abs(I_batt) * self.Battery.timestep) / 3600.0
        Ic_rate = np.abs(I_batt) / self.Battery.Cn
        c1 = cache["c1_states"][ids][:, None]
        c2 = cache["c2_states"][ids][:, None]
        q_loss_mAh = (
            (c1 * soc_arr + c2)
            * np.exp((-31700.0 + 163.3 * Ic_rate) * self.Battery._loss_inv_rt)
            * (current_step_Ah ** self.Battery._loss_z)
        )
        dsoh_batt = (q_loss_mAh * self.Battery._loss_mAh_to_Ah) / self.Battery.Cn

        soc_low, soc_high = self.soc_bounds
        soc_err = (np.abs(soc_new - self.SOC_target)) ** 2
        w_soc = np.where((soc_new >= soc_high) | (soc_new <= soc_low), self.w_soc * 10, self.w_soc)
        soc_cost = w_soc * soc_err
        h2_cost = self.w_h2 * cache["h2_equal_ta"][int(time_step)][None, :]
        fcs_soh_cost = self.w_fc * cache["dsoh_fcs"][None, :]
        batt_soh_cost = self.w_batt * dsoh_batt
        objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost
        reward = -objective_cost
        return soc_new, reward
        
    def execute(self, action, car_spd, car_acc, soc):
        action_value = float(np.asarray(action, dtype=np.float64).reshape(()))
        soc_value = float(np.asarray(soc, dtype=np.float64).reshape(()))
        action_mapped = np.clip((action_value + 1.0) / 2.0, 0.0, 1.0)
        P_FCS = action_mapped * self.FCHEV.P_FC_max  # kW
        T_axle, W_axle, P_axle = self.FCHEV.T_W_axle(car_spd, car_acc)
        T_mot, W_mot, mot_eff, P_mot = self.FCHEV.run_motor(T_axle, W_axle, P_axle)  # W
        P_dcdc, h2_fcs, info_fcs = self.FCHEV.run_fuel_cell(P_FCS)  # kW
        P_dcdc = float(np.asarray(P_dcdc, dtype=np.float64).reshape(()))
        h2_fcs = float(np.asarray(h2_fcs, dtype=np.float64).reshape(()))
        dsoh_fcs, info_fcs_soh = self.FCHEV.run_FC_SOH(float(P_FCS))
        self.P_batt = float(np.clip(
            P_mot - P_dcdc * 1000.0,
            -self.Battery.batt_maxpower,
            self.Battery.batt_maxpower,
        ))

        Voc = float(np.asarray(self.Battery.ocv_func(soc_value), dtype=np.float64).reshape(()))
        r0 = float(np.asarray(self.Battery.r0_func(soc_value), dtype=np.float64).reshape(()))
        delta = Voc ** 2 - 4.0 * r0 * self.P_batt
        if delta < 0.0:
            I_batt = Voc / (2.0 * r0)
        else:
            I_batt = (Voc - np.sqrt(delta)) / (2.0 * r0)
        soc_deriv = self.Battery.timestep * (I_batt / 3600.0 / self.Battery.Cn)
        SOC_new = float(np.clip(soc_value - soc_deriv, self.Battery.soc_min, self.Battery.soc_max))
        Voc_new = float(np.asarray(self.Battery.ocv_func(SOC_new), dtype=np.float64).reshape(()))
        Vt_new = Voc_new - r0 * I_batt
        power_out = Vt_new * I_batt

        current_step_Ah = (np.abs(I_batt) * self.Battery.timestep) / 3600.0
        Ic_rate = np.abs(I_batt) / self.Battery.Cn
        if soc_value <= self.Battery._loss_soc_split:
            c1 = self.Battery._loss_c1_low
            c2 = self.Battery._loss_c2_low
        else:
            c1 = self.Battery._loss_c1_high
            c2 = self.Battery._loss_c2_high
        q_loss_mAh = ((c1 * soc_value + c2)
                      * np.exp((-31700.0 + 163.3 * Ic_rate) * self.Battery._loss_inv_rt)
                      * (current_step_Ah ** self.Battery._loss_z))
        dsoh = float((q_loss_mAh * self.Battery._loss_mAh_to_Ah) / self.Battery.Cn)

        SOH_new = 1.0 - dsoh
        info_batt = {'SOC': SOC_new, 'SOH': SOH_new, 'soc_deriv': soc_deriv,
                     'pack_OCV': Voc_new, 'pack_Vt': Vt_new,
                     'I': I_batt, 'I_c': Ic_rate, 'pack_power_out': power_out/1000.0,
                     'P_batt_req': self.P_batt / 1000.0,
                     'tep_a': self.Battery._loss_t_env,
                     'dsoh': dsoh}
        self.dSOH_batt = dsoh
        self.info = {}
        self.info.update({'T_axle': T_axle, 'W_axle': W_axle, 'P_axle': P_axle/1000,
                          'T_mot': T_mot, 'W_mot': W_mot, 'mot_eff': mot_eff,
                          'P_mot': P_mot/1000})
        self.info.update(info_fcs)
        self.info.update(info_batt)
        self.info.update(info_fcs_soh)
        # SOC-NEW
        self.SOC = SOC_new
        self.h2_fcs = h2_fcs
        self.h2_batt = max(0.0, self.P_batt) / 1000.0 * self.eq_h2_batt_coef
        self.h2_equal = self.h2_fcs + self.h2_batt
        self.dSOH_FCS = float(dsoh_fcs)
        return self.SOC

    def _equivalent_hydrogen_terms(self):
        h2_fcs = float(self.h2_fcs)
        if not np.isfinite(h2_fcs) or h2_fcs < 0.0:
            h2_fcs = 0.0
        p_batt = float(self.P_batt)
        if not np.isfinite(p_batt):
            p_batt = 0.0
        h2_batt = max(0.0, p_batt) / 1000.0 * self.eq_h2_batt_coef
        h2_equal = h2_fcs + h2_batt
        return h2_fcs, h2_batt, h2_equal
    
    def get_reward(self):
        soc = float(self.SOC)
        soc_low, soc_high = self.soc_bounds
        # SOC deviation from target
        soc_err = (np.abs(soc - self.SOC_target))**2
        #w_soc = self.w_soc
        w_soc = np.where((soc >= soc_high) | (soc <= soc_low), self.w_soc * 10, self.w_soc)
        soc_cost = w_soc * (soc_err)

        h2_fcs, h2_batt, h2_equal = self._equivalent_hydrogen_terms()
        self.h2_batt = h2_batt
        self.h2_equal = h2_equal
        h2_cost = self.w_h2 * h2_equal
        # Cost of fuel cell degradation (scaled)
        fcs_soh_cost = self.w_fc * float(self.dSOH_FCS)
        # Cost of battery degradation (scaled)
        batt_soh_cost = self.w_batt * float(self.dSOH_batt)
        
        in_bounds = (soc_low < soc < soc_high)
        objective_cost = soc_cost + h2_cost + fcs_soh_cost + batt_soh_cost
        reward = -objective_cost
        reward = float(reward)

        self.info.update({'EMS_reward': reward,
                          'h2_fcs': float(h2_fcs),
                          'h2_batt': float(h2_batt),
                          'h2_equal': float(h2_equal),
                          'soc_cost': float(soc_cost),
                          'h2_cost': float(h2_cost),
                          'fcs_soh_cost': float(fcs_soh_cost),
                          'batt_soh_cost': float(batt_soh_cost),
                          'objective_cost': float(objective_cost),
                          'soc_in_bounds': int(in_bounds)
})
        return reward

    def get_info(self):
        return self.info
