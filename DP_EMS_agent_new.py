import numpy as np

from FCHEV_SOH import FCHEV_SOH
from Battery import CellModel1


class DP_EMS_agent_new:
    """
    DP EMS agent with an augmented backward-DP state:
    state = (SOC, previous fuel-cell power).

    The existing DP_EMS_agent only discretizes SOC for the backward pass, so the
    FC cycling degradation term is evaluated against a single global
    `self.FCHEV.P_FC_old`. That makes the Bellman stage cost inconsistent because
    FC degradation actually depends on the transition
    `previous FC power -> current FC power(action)`.

    This agent keeps the forward execution interface compatible with the current
    code (`execute`, `get_reward`, `get_info`) and adds helper methods for a
    joint-state backward DP:
    - `prepare_backward_cache(..., prev_fc_states=...)`
    - `batch_transition_reward_with_prev_fc(...)`
    - `batch_transition_reward_augmented(...)`
    - `flatten_joint_state_ids(...)` / `unflatten_joint_state_ids(...)`
    """

    def __init__(
        self,
        w_soc=100,
        soc0=0.6,
        SOC_MODE="CS",
        soc_ref=1.0,
        w_h2=10.0,
        w_fc=4.275e6,
        w_batt=1.116e6,
        eq_h2_batt_coef=0.0164,
        soc_target=0.6,
        soc_bounds=(0.4, 0.8),
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

        if soc_target is not None:
            self.SOC_target = soc_target
        elif SOC_MODE == "CD":
            self.SOC_target = soc0 - 0.2
        else:
            self.SOC_target = soc0

        self.SOC = self.SOC_target
        self.h2_fcs = 0.0
        self.h2_batt = 0.0
        self.h2_equal = 0.0
        self.dSOH_FCS = 0.0
        self.P_batt = 0.0
        self.dSOH_batt = 0.0
        self._bw_cache = None

    @staticmethod
    def _as_float_array(values):
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _as_int_array(values):
        return np.asarray(values, dtype=np.int64)

    def action_to_fc_power(self, actions):
        actions_arr = self._as_float_array(actions)
        actions_mapped = np.clip((actions_arr + 1.0) / 2.0, 0.0, 1.0)
        return actions_mapped * self.FCHEV.P_FC_max

    def build_prev_fc_grid(self, actions=None, prev_fc_states=None):
        if prev_fc_states is not None:
            grid = self._as_float_array(prev_fc_states).reshape(-1)
        else:
            if actions is None:
                raise ValueError("actions must be provided when prev_fc_states is None")
            grid = self.action_to_fc_power(actions).reshape(-1)
            grid = np.concatenate(([0.0], grid))
        grid = np.unique(np.round(grid.astype(np.float64), decimals=12))
        grid.sort()
        return grid

    def build_uniform_prev_fc_grid(self, state_count):
        state_count = max(2, int(state_count))
        grid = np.linspace(0.0, self.FCHEV.P_FC_max, state_count, dtype=np.float64)
        grid[0] = 0.0
        grid[-1] = self.FCHEV.P_FC_max
        return self.build_prev_fc_grid(prev_fc_states=grid)

    def _compute_fc_soh_batch(self, P_FCS, P_fc_old):
        P_FCS_arr = self._as_float_array(P_FCS)
        P_fc_old_arr = self._as_float_array(P_fc_old)

        P_fc_low = self.FCHEV.P_FC_low
        P_fc_high = self.FCHEV.P_FC_high
        P_fc_off = self.FCHEV.P_FC_off
        simtime = self.FCHEV.simtime

        on_old = P_fc_old_arr >= P_fc_off
        on_new = P_FCS_arr >= P_fc_off

        d_s_s = np.where((~on_old) & on_new, simtime * 1.96, 0.0)
        d_low = np.where(on_new & (P_FCS_arr < P_fc_low), simtime * 1.26, 0.0)
        d_high = np.where(P_FCS_arr >= P_fc_high, simtime * 1.47, 0.0)
        d_l_c = 5.93 * self.FCHEV.load_change_cycle_count(P_FCS_arr, P_fc_old_arr)
        De_i = (d_s_s + d_low + d_high) * 1e-3 / 3600 + d_l_c * 1e-5
        return De_i / 100.0

    def _resolve_soc_grid_meta(self, soc_states):
        soc_states = self._as_float_array(soc_states).reshape(-1)
        if soc_states.size <= 1:
            return float(soc_states[0]), 1.0, True
        soc_inc = float(soc_states[1] - soc_states[0])
        is_uniform = bool(np.allclose(np.diff(soc_states), soc_inc))
        return float(soc_states[0]), soc_inc, is_uniform

    @staticmethod
    def _nearest_grid_ids(grid, values):
        grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        values = np.asarray(values, dtype=np.float64)
        orig_shape = values.shape
        flat = values.reshape(-1)

        if grid.size == 1:
            ids = np.zeros(flat.shape, dtype=np.int64)
            return ids.reshape(orig_shape)

        insert_ids = np.searchsorted(grid, flat)
        insert_ids = np.clip(insert_ids, 1, grid.size - 1)
        left_ids = insert_ids - 1
        right_ids = insert_ids
        left = grid[left_ids]
        right = grid[right_ids]
        choose_right = np.abs(flat - right) < np.abs(flat - left)
        ids = np.where(choose_right, right_ids, left_ids).astype(np.int64)
        return ids.reshape(orig_shape)

    def map_prev_fc_powers_to_state_ids(self, prev_fc_powers):
        if self._bw_cache is None:
            raise RuntimeError("backward cache is not prepared; call prepare_backward_cache() first")

        grid = self._bw_cache["prev_fc_states"]
        return self._nearest_grid_ids(grid, prev_fc_powers)

    def soc_to_state_ids(self, soc_values):
        if self._bw_cache is None:
            raise RuntimeError("backward cache is not prepared; call prepare_backward_cache() first")

        soc_states = self._bw_cache["soc_states"]
        soc_values = self._as_float_array(soc_values)
        orig_shape = soc_values.shape
        flat = soc_values.reshape(-1)

        if self._bw_cache["soc_grid_is_uniform"]:
            ids = np.rint((flat - self._bw_cache["soc_grid_min"]) / self._bw_cache["soc_grid_inc"]).astype(np.int64)
            ids = np.clip(ids, 0, soc_states.size - 1)
            return ids.reshape(orig_shape)

        insert_ids = np.searchsorted(soc_states, flat)
        insert_ids = np.clip(insert_ids, 1, soc_states.size - 1)
        left_ids = insert_ids - 1
        right_ids = insert_ids
        left = soc_states[left_ids]
        right = soc_states[right_ids]
        choose_right = np.abs(flat - right) < np.abs(flat - left)
        ids = np.where(choose_right, right_ids, left_ids).astype(np.int64)
        return ids.reshape(orig_shape)

    def flatten_joint_state_ids(self, soc_state_ids, prev_fc_state_ids):
        if self._bw_cache is None:
            raise RuntimeError("backward cache is not prepared; call prepare_backward_cache() first")

        soc_ids = self._as_int_array(soc_state_ids)
        prev_ids = self._as_int_array(prev_fc_state_ids)
        if soc_ids.shape != prev_ids.shape:
            raise ValueError("soc_state_ids and prev_fc_state_ids must have the same shape")
        prev_fc_count = int(self._bw_cache["prev_fc_states"].size)
        return soc_ids * prev_fc_count + prev_ids

    def unflatten_joint_state_ids(self, joint_state_ids):
        if self._bw_cache is None:
            raise RuntimeError("backward cache is not prepared; call prepare_backward_cache() first")

        joint_ids = self._as_int_array(joint_state_ids)
        prev_fc_count = int(self._bw_cache["prev_fc_states"].size)
        soc_ids = joint_ids // prev_fc_count
        prev_ids = joint_ids % prev_fc_count
        return soc_ids, prev_ids

    def joint_state_count(self):
        if self._bw_cache is None:
            raise RuntimeError("backward cache is not prepared; call prepare_backward_cache() first")
        return int(self._bw_cache["soc_states"].size * self._bw_cache["prev_fc_states"].size)

    def initial_prev_fc_state_id(self):
        if self._bw_cache is None:
            raise RuntimeError("backward cache is not prepared; call prepare_backward_cache() first")
        return int(self.map_prev_fc_powers_to_state_ids(np.array([0.0], dtype=np.float64))[0])

    def build_terminal_value_table(self, soc_target=None):
        if self._bw_cache is None:
            raise RuntimeError("backward cache is not prepared; call prepare_backward_cache() first")

        if soc_target is None:
            soc_target = self.SOC_target
        soc_target = float(soc_target)

        soc_states = self._bw_cache["soc_states"]
        prev_fc_states = self._bw_cache["prev_fc_states"]
        terminal = np.full((soc_states.size, prev_fc_states.size), -np.inf, dtype=np.float32)
        target_soc_id = int(self.soc_to_state_ids(np.array([soc_target], dtype=np.float64))[0])
        terminal[target_soc_id, :] = 0.0
        return terminal

    def prepare_backward_cache(self, actions, speed_list, acc_list, states, prev_fc_states=None):
        actions_arr = self._as_float_array(actions).reshape(-1)
        speed_arr = self._as_float_array(speed_list).reshape(-1)
        acc_arr = self._as_float_array(acc_list).reshape(-1)
        soc_states = self._as_float_array(states).reshape(-1)
        prev_fc_grid = self.build_prev_fc_grid(actions=actions_arr, prev_fc_states=prev_fc_states)
        soc_grid_min, soc_grid_inc, soc_grid_is_uniform = self._resolve_soc_grid_meta(soc_states)

        P_FCS = self.action_to_fc_power(actions_arr).reshape(-1)
        P_dcdc, h2_fcs, _ = self.FCHEV.run_fuel_cell(P_FCS)
        P_dcdc = self._as_float_array(P_dcdc).reshape(-1)
        h2_fcs = self._as_float_array(h2_fcs).reshape(-1)

        dsoh_fcs_legacy = self._compute_fc_soh_batch(P_FCS, float(self.FCHEV.P_FC_old)).reshape(-1)
        dsoh_fcs_prev_action = self._compute_fc_soh_batch(P_FCS[None, :], prev_fc_grid[:, None]).astype(np.float64)
        next_prev_fc_ids = self._nearest_grid_ids(prev_fc_grid, P_FCS)

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
        h2_batt_ta = P_batt_ta / 1000.0 * self.eq_h2_batt_coef
        h2_equal_ta = h2_fcs[None, :] + h2_batt_ta

        ocv_states = self._as_float_array(self.Battery.ocv_func(soc_states)).reshape(-1)
        r0_states = self._as_float_array(self.Battery.r0_func(soc_states)).reshape(-1)
        low_mask = soc_states <= self.Battery._loss_soc_split
        c1_states = np.where(low_mask, self.Battery._loss_c1_low, self.Battery._loss_c1_high)
        c2_states = np.where(low_mask, self.Battery._loss_c2_low, self.Battery._loss_c2_high)

        self._bw_cache = {
            "actions": actions_arr,
            "soc_states": soc_states,
            "prev_fc_states": prev_fc_grid,
            "soc_grid_min": soc_grid_min,
            "soc_grid_inc": soc_grid_inc,
            "soc_grid_is_uniform": soc_grid_is_uniform,
            "P_FCS": P_FCS,
            "next_prev_fc_ids": next_prev_fc_ids.astype(np.int64),
            "h2_fcs": h2_fcs,
            "dsoh_fcs_legacy": dsoh_fcs_legacy,
            "dsoh_fcs_prev_action": dsoh_fcs_prev_action,
            "p_batt_ta": P_batt_ta,
            "h2_batt_ta": h2_batt_ta,
            "h2_equal_ta": h2_equal_ta,
            "ocv_states": ocv_states,
            "r0_states": r0_states,
            "c1_states": c1_states,
            "c2_states": c2_states,
        }
        return self._bw_cache

    def _batch_step_terms(self, time_step, soc_state_ids):
        if self._bw_cache is None:
            raise RuntimeError("backward cache is not prepared; call prepare_backward_cache() first")

        ids = self._as_int_array(soc_state_ids).reshape(-1)
        if ids.size == 0:
            empty = np.empty((0, self._bw_cache["actions"].shape[0]), dtype=np.float64)
            return empty, empty, empty, empty

        cache = self._bw_cache
        p_batt = cache["p_batt_ta"][int(time_step)][None, :]

        soc_arr = cache["soc_states"][ids][:, None]
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
        batt_soh_cost = self.w_batt * dsoh_batt
        base_cost = soc_cost + h2_cost + batt_soh_cost
        return soc_new, base_cost, soc_cost, batt_soh_cost

    def batch_transition_reward(self, time_step, state_ids):
        """
        Backward-compatible API for the legacy 1D-SOC DP.

        FC degradation is evaluated against a single global `self.FCHEV.P_FC_old`,
        matching the behavior of the original DP_EMS_agent.
        """
        soc_new, base_cost, _, _ = self._batch_step_terms(time_step, state_ids)
        if soc_new.size == 0:
            return soc_new, base_cost

        fcs_soh_cost = self.w_fc * self._bw_cache["dsoh_fcs_legacy"][None, :]
        objective_cost = base_cost + fcs_soh_cost
        reward = -objective_cost
        return soc_new, reward

    def batch_transition_reward_with_prev_fc(self, time_step, soc_state_ids, prev_fc_state_ids):
        """
        Augmented-state backward transition:
        input state = (soc_state_id, prev_fc_state_id)
        next state = (next_soc_state_id, action-matched next_prev_fc_state_id)
        """
        soc_ids = self._as_int_array(soc_state_ids).reshape(-1)
        prev_ids = self._as_int_array(prev_fc_state_ids).reshape(-1)
        if soc_ids.size == 0:
            empty_float = np.empty((0, self._bw_cache["actions"].shape[0]), dtype=np.float64)
            empty_int = np.empty((0, self._bw_cache["actions"].shape[0]), dtype=np.int64)
            return empty_float, empty_int, empty_float
        if prev_ids.size == 1 and soc_ids.size > 1:
            prev_ids = np.full(soc_ids.shape, int(prev_ids[0]), dtype=np.int64)
        if prev_ids.shape != soc_ids.shape:
            raise ValueError("soc_state_ids and prev_fc_state_ids must have the same shape")

        soc_new, base_cost, _, _ = self._batch_step_terms(time_step, soc_ids)
        fcs_soh_cost = self.w_fc * self._bw_cache["dsoh_fcs_prev_action"][prev_ids, :]
        objective_cost = base_cost + fcs_soh_cost
        reward = -objective_cost

        next_prev_fc_ids = np.broadcast_to(
            self._bw_cache["next_prev_fc_ids"][None, :],
            reward.shape,
        ).astype(np.int64, copy=False)
        return soc_new, next_prev_fc_ids, reward

    def get_fc_cycle_reward_matrix(self, prev_fc_state_ids=None):
        """
        Return FC cycling reward contribution only:
            reward_fc(prev_fc, action) = - w_fc * dSOH_fc(prev_fc -> action)
        """
        if self._bw_cache is None:
            raise RuntimeError("backward cache is not prepared; call prepare_backward_cache() first")

        cycle_reward = -self.w_fc * self._bw_cache["dsoh_fcs_prev_action"]
        if prev_fc_state_ids is None:
            return cycle_reward
        prev_ids = self._as_int_array(prev_fc_state_ids).reshape(-1)
        return cycle_reward[prev_ids, :]

    def batch_soc_action_base_reward(self, time_step, soc_state_ids):
        """
        Return the part of the Bellman one-step reward that depends only on
        SOC-state and action, not on previous FC power.

        Outputs:
            next_soc_ids: [num_soc_states, num_actions]
            next_prev_fc_ids: [num_soc_states, num_actions]
            base_reward: [num_soc_states, num_actions]
                = -(soc_cost + h2_cost + batt_soh_cost)
        """
        soc_new, base_cost, _, _ = self._batch_step_terms(time_step, soc_state_ids)
        next_soc_ids = self.soc_to_state_ids(soc_new)
        next_prev_fc_ids = np.broadcast_to(
            self._bw_cache["next_prev_fc_ids"][None, :],
            base_cost.shape,
        ).astype(np.int64, copy=False)
        base_reward = -base_cost
        return next_soc_ids, next_prev_fc_ids, base_reward

    def batch_transition_reward_augmented(self, time_step, joint_state_ids):
        """
        Convenience wrapper for flattened joint-state indexing.

        Returns:
            next_joint_state_ids: shape [num_joint_states, num_actions]
            reward: shape [num_joint_states, num_actions]
        """
        soc_ids, prev_ids = self.unflatten_joint_state_ids(joint_state_ids)
        soc_new, next_prev_fc_ids, reward = self.batch_transition_reward_with_prev_fc(
            time_step=time_step,
            soc_state_ids=soc_ids,
            prev_fc_state_ids=prev_ids,
        )
        next_soc_ids = self.soc_to_state_ids(soc_new)
        next_joint_ids = self.flatten_joint_state_ids(next_soc_ids, next_prev_fc_ids)
        return next_joint_ids, reward

    def reset_forward_state(self, soc=None, prev_fc_power=0.0):
        if soc is None:
            soc = self.SOC_target
        self.SOC = float(soc)
        self.h2_fcs = 0.0
        self.h2_batt = 0.0
        self.h2_equal = 0.0
        self.dSOH_FCS = 0.0
        self.P_batt = 0.0
        self.dSOH_batt = 0.0
        self.info = {}
        self.FCHEV.P_FC_old = float(prev_fc_power)

    def execute(self, action, car_spd, car_acc, soc):
        action_value = float(np.asarray(action, dtype=np.float64).reshape(()))
        soc_value = float(np.asarray(soc, dtype=np.float64).reshape(()))
        P_FCS = float(self.action_to_fc_power(np.array([action_value], dtype=np.float64))[0])
        prev_fc_power = float(self.FCHEV.P_FC_old)

        T_axle, W_axle, P_axle = self.FCHEV.T_W_axle(car_spd, car_acc)
        T_mot, W_mot, mot_eff, P_mot = self.FCHEV.run_motor(T_axle, W_axle, P_axle)
        P_dcdc, h2_fcs, info_fcs = self.FCHEV.run_fuel_cell(P_FCS)
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
        q_loss_mAh = (
            (c1 * soc_value + c2)
            * np.exp((-31700.0 + 163.3 * Ic_rate) * self.Battery._loss_inv_rt)
            * (current_step_Ah ** self.Battery._loss_z)
        )
        dsoh = float((q_loss_mAh * self.Battery._loss_mAh_to_Ah) / self.Battery.Cn)

        SOH_new = 1.0 - dsoh
        info_batt = {
            "SOC": SOC_new,
            "SOH": SOH_new,
            "soc_deriv": soc_deriv,
            "pack_OCV": Voc_new,
            "pack_Vt": Vt_new,
            "I": I_batt,
            "I_c": Ic_rate,
            "pack_power_out": power_out / 1000.0,
            "P_batt_req": self.P_batt / 1000.0,
            "tep_a": self.Battery._loss_t_env,
            "dsoh": dsoh,
        }

        self.dSOH_batt = dsoh
        self.info = {}
        self.info.update({
            "T_axle": T_axle,
            "W_axle": W_axle,
            "P_axle": P_axle / 1000.0,
            "T_mot": T_mot,
            "W_mot": W_mot,
            "mot_eff": mot_eff,
            "P_mot": P_mot / 1000.0,
        })
        self.info.update(info_fcs)
        self.info.update(info_batt)
        self.info.update(info_fcs_soh)
        self.info.update({
            "prev_fc_power": prev_fc_power,
            "current_fc_power": float(P_FCS),
        })

        self.SOC = SOC_new
        self.h2_fcs = h2_fcs
        self.h2_batt =  (self.P_batt / 1000.0) * self.eq_h2_batt_coef
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
        h2_batt = (p_batt / 1000.0) * self.eq_h2_batt_coef
        h2_equal = h2_fcs + h2_batt
        return h2_fcs, h2_batt, h2_equal

    def get_reward(self):
        soc = float(self.SOC)
        soc_low, soc_high = self.soc_bounds
        soc_err = (np.abs(soc - self.SOC_target)) ** 2
        w_soc = np.where((soc >= soc_high) | (soc <= soc_low), self.w_soc * 10, self.w_soc)
        soc_cost = w_soc * soc_err

        h2_fcs, h2_batt, h2_equal = self._equivalent_hydrogen_terms()
        self.h2_batt = h2_batt
        self.h2_equal = h2_equal
        h2_cost = self.w_h2 * h2_fcs
        fcs_soh_cost = self.w_fc * float(self.dSOH_FCS)
        batt_soh_cost = self.w_batt * float(self.dSOH_batt)

        in_bounds = (soc_low < soc < soc_high)
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
            "soc_in_bounds": int(in_bounds),
        })
        return reward

    def get_info(self):
        return self.info


# Convenience alias so downstream code can simply switch the import path.
DP_EMS_agent = DP_EMS_agent_new
