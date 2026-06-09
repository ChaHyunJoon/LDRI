"""
Dynamic Programming, energy management

2D DP state:
    x_t = (SOC_t, previous_FC_power_t)

Compared with the legacy 1D DP over SOC only, this solver uses the augmented
state needed to evaluate FC cycling degradation consistently.

Implementation notes:
- Backward value recursion keeps only two time slices in RAM.
- The reward-optimal policy is stored as a disk-backed memmap `.npy` file to
  avoid allocating a full 3D table in memory.
- Backward computation is chunked over SOC states to bound temporary matrix
  sizes.
- The Bellman backup is factorized so SOC-dependent terms are computed once per
  `(SOC, action)` pair and FC-cycling terms are added separately for each
  `previous_FC_power` state.
"""
import os
import time

import numpy as np
import scipy.io as scio
from tqdm import tqdm

from common.arguments import get_args
from DP_env import DP_Env
from DP_EMS_agent_new import DP_EMS_agent_new


class DP_brain_2D:
    def __init__(
        self,
        env,
        DP_EMS_Agent,
        output_dir,
        gamma=0.99,
        soc_chunk_size=128,
        prev_fc_state_count=0,
    ):
        self.DP_EMS_agent = DP_EMS_Agent
        self.env = env
        self.gamma = float(gamma)
        self.soc_chunk_size = max(1, int(soc_chunk_size))
        self.prev_fc_state_count = int(prev_fc_state_count)

        self.states = self.env.states.astype(np.float64)
        self.actions = self.env.actions.astype(np.float64)
        if self.prev_fc_state_count > 0:
            self.prev_fc_states = self.DP_EMS_agent.build_uniform_prev_fc_grid(self.prev_fc_state_count)
        else:
            self.prev_fc_states = self.DP_EMS_agent.build_prev_fc_grid(actions=self.actions)

        self.soc_count = int(self.states.size)
        self.prev_fc_count = int(self.prev_fc_states.size)
        self.action_count = int(self.actions.size)
        self.time_steps = int(self.env.time_steps)
        self.soc_grid_tol = max(1e-12, 0.5 * float(self.env.state_increment))

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.action_id_dtype = np.uint8 if self.action_count < np.iinfo(np.uint8).max else np.uint16
        self.invalid_action_id = np.iinfo(self.action_id_dtype).max
        self.fc_cycle_reward_matrix = None

        self.policy_reward_action_path = os.path.join(self.output_dir, "reward_action_id.npy")
        self.policy_reward_action_id = np.lib.format.open_memmap(
            self.policy_reward_action_path,
            mode="w+",
            dtype=self.action_id_dtype,
            shape=(self.time_steps - 1, self.soc_count, self.prev_fc_count),
        )

        self.value_curr = np.full((self.soc_count, self.prev_fc_count), -np.inf, dtype=np.float32)
        self.value_next = np.full_like(self.value_curr, -np.inf)
        self.value_t0 = None

        self.policy_action_value = np.full(self.time_steps - 1, np.nan, dtype=np.float32)
        self.policy_action_index = np.full(self.time_steps - 1, -1, dtype=np.int32)
        self.policy_prev_fc_power = np.full(self.time_steps - 1, np.nan, dtype=np.float32)
        self.policy_prev_fc_state = np.full(self.time_steps - 1, -1, dtype=np.int32)

        self.info_dict = {
            "T_mot": [],
            "W_mot": [],
            "mot_eff": [],
            "P_mot": [],
            "P_fc": [],
            "P_fce": [],
            "fce_eff": [],
            "FCS_SOH": [],
            "P_dcdc": [],
            "dcdc_eff": [],
            "FCS_De": [],
            "travel": [],
            "d_s_s": [],
            "d_low": [],
            "d_high": [],
            "d_l_c": [],
            "speed": [],
            "acc": [],
            "EMS_reward": [],
            "soc_cost": [],
            "h2_fcs": [],
            "h2_batt": [],
            "h2_equal": [],
            "objective_cost": [],
            "h2_cost": [],
            "batt_soh_cost": [],
            "fcs_soh_cost": [],
            "SOC": [],
            "SOH": [],
            "I": [],
            "I_c": [],
            "pack_OCV": [],
            "pack_Vt": [],
            "pack_power_out": [],
            "P_batt_req": [],
            "tep_a": [],
            "dsoh": [],
            "prev_fc_power": [],
            "current_fc_power": [],
            "prev_fc_state_id": [],
            "current_fc_state_id": [],
        }

    def _soc_reachability_window(self, soc_target):
        terminal_step = self.time_steps - 1
        soc_init = float(self.env.state_init)
        max_delta_soc = 0.01
        min_soc = float(np.min(self.states))
        max_soc = float(np.max(self.states))

        min_reachable_soc = np.zeros(self.time_steps, dtype=np.float32)
        max_reachable_soc = np.zeros(self.time_steps, dtype=np.float32)
        min_reachable_soc[0] = soc_init
        max_reachable_soc[0] = soc_init
        for t in range(1, self.time_steps):
            min_reachable_soc[t] = min_reachable_soc[t - 1] - max_delta_soc
            max_reachable_soc[t] = max_reachable_soc[t - 1] + max_delta_soc
        min_reachable_soc = np.clip(min_reachable_soc, min_soc, max_soc)
        max_reachable_soc = np.clip(max_reachable_soc, min_soc, max_soc)

        min_to_target_soc = np.zeros(self.time_steps, dtype=np.float32)
        max_to_target_soc = np.zeros(self.time_steps, dtype=np.float32)
        for t in range(self.time_steps):
            remaining_steps = terminal_step - t
            min_to_target_soc[t] = soc_target - remaining_steps * max_delta_soc
            max_to_target_soc[t] = soc_target + remaining_steps * max_delta_soc
        min_to_target_soc = np.clip(min_to_target_soc, min_soc, max_soc)
        max_to_target_soc = np.clip(max_to_target_soc, min_soc, max_soc)
        return min_reachable_soc, max_reachable_soc, min_to_target_soc, max_to_target_soc

    def _soc_valid_ids_at_time(self, time_step, min_reachable_soc, max_reachable_soc, min_to_target_soc, max_to_target_soc):
        tol = self.soc_grid_tol
        valid_mask = (
            (self.states >= min_reachable_soc[time_step] - tol)
            & (self.states <= max_reachable_soc[time_step] + tol)
            & (self.states >= min_to_target_soc[time_step] - tol)
            & (self.states <= max_to_target_soc[time_step] + tol)
        )
        return np.flatnonzero(valid_mask).astype(np.int64)

    def find_soc_idx(self, soc_value):
        idx = int(self.DP_EMS_agent.soc_to_state_ids(np.array([soc_value], dtype=np.float64))[0])
        return idx, float(self.states[idx])

    def find_prev_fc_idx(self, prev_fc_power):
        idx = int(self.DP_EMS_agent.map_prev_fc_powers_to_state_ids(np.array([prev_fc_power], dtype=np.float64))[0])
        return idx, float(self.prev_fc_states[idx])

    def execute(self, car_spd, car_acc, action, soc):
        soc_new = self.DP_EMS_agent.execute(
            action=action,
            car_spd=car_spd,
            car_acc=car_acc,
            soc=soc,
        )
        self.DP_EMS_agent.get_reward()
        return soc_new, self.DP_EMS_agent.get_info()

    def save_backward_artifacts(self):
        self.policy_reward_action_id.flush()
        np.save(os.path.join(self.output_dir, "soc_states.npy"), self.states)
        np.save(os.path.join(self.output_dir, "prev_fc_states.npy"), self.prev_fc_states)
        np.save(os.path.join(self.output_dir, "actions.npy"), self.actions)
        if self.value_t0 is not None:
            np.save(os.path.join(self.output_dir, "value_t0.npy"), self.value_t0)
        scio.savemat(
            os.path.join(self.output_dir, "dp_2d_meta.mat"),
            mdict={
                "soc_states": self.states,
                "prev_fc_states": self.prev_fc_states,
                "actions": self.actions,
                "value_t0": np.asarray(self.value_t0) if self.value_t0 is not None else np.empty((0, 0), dtype=np.float32),
            },
        )

    def DP_backward(self):
        terminal_step = self.time_steps - 1
        soc_target = float(getattr(self.DP_EMS_agent, "SOC_target", 0.6))
        self.DP_EMS_agent.SOC_target = soc_target

        cache = self.DP_EMS_agent.prepare_backward_cache(
            actions=self.actions,
            speed_list=self.env.speed_list,
            acc_list=self.env.acc_list,
            states=self.states,
            prev_fc_states=self.prev_fc_states,
        )
        self.prev_fc_states = cache["prev_fc_states"]
        self.prev_fc_count = int(self.prev_fc_states.size)
        self.fc_cycle_reward_matrix = self.DP_EMS_agent.get_fc_cycle_reward_matrix().astype(np.float32, copy=False)

        if self.policy_reward_action_id.shape[2] != self.prev_fc_count:
            raise RuntimeError("policy storage prev_fc dimension does not match prepared cache")

        self.value_next[:, :] = self.DP_EMS_agent.build_terminal_value_table(soc_target=soc_target).astype(np.float32)
        self.value_curr[:, :] = -np.inf

        min_reachable_soc, max_reachable_soc, min_to_target_soc, max_to_target_soc = self._soc_reachability_window(soc_target)
        all_prev_ids = np.arange(self.prev_fc_count, dtype=np.int64)
        initial_prev_fc_id = int(self.DP_EMS_agent.initial_prev_fc_state_id())

        for time_step in tqdm(range(self.time_steps - 2, -1, -1)):
            self.value_curr[:, :] = -np.inf
            self.policy_reward_action_id[time_step, :, :] = self.invalid_action_id

            valid_soc_ids = self._soc_valid_ids_at_time(
                time_step,
                min_reachable_soc,
                max_reachable_soc,
                min_to_target_soc,
                max_to_target_soc,
            )
            if time_step == 0 and valid_soc_ids.size == 0:
                init_soc_id = int(self.DP_EMS_agent.soc_to_state_ids(np.array([self.env.state_init], dtype=np.float64))[0])
                valid_soc_ids = np.array([init_soc_id], dtype=np.int64)
            if valid_soc_ids.size == 0:
                self.value_curr, self.value_next = self.value_next, self.value_curr
                continue

            if time_step == 0:
                prev_ids_eval = np.array([initial_prev_fc_id], dtype=np.int64)
            else:
                prev_ids_eval = all_prev_ids

            for start in range(0, valid_soc_ids.size, self.soc_chunk_size):
                soc_chunk = valid_soc_ids[start:start + self.soc_chunk_size]
                next_soc_ids, next_prev_fc_ids, base_reward = self.DP_EMS_agent.batch_soc_action_base_reward(
                    time_step=time_step,
                    soc_state_ids=soc_chunk,
                )
                next_value = self.value_next[next_soc_ids, next_prev_fc_ids]

                feasible = np.isfinite(next_value)
                candidate_base = np.where(feasible, base_reward + self.gamma * next_value, -np.inf).astype(
                    np.float32,
                    copy=False,
                )

                cycle_reward = self.fc_cycle_reward_matrix[prev_ids_eval, :]
                total_reward = candidate_base[:, None, :] + cycle_reward[None, :, :]
                finite_mask = np.isfinite(total_reward)
                row_has_finite = np.any(finite_mask, axis=2)
                if not np.any(row_has_finite):
                    continue

                best_action_id = np.argmax(total_reward, axis=2).astype(np.int64)
                best_reward = np.max(total_reward, axis=2)

                soc_grid = np.broadcast_to(soc_chunk[:, None], best_action_id.shape)
                prev_grid = np.broadcast_to(prev_ids_eval[None, :], best_action_id.shape)

                keep_soc_ids = soc_grid[row_has_finite]
                keep_prev_ids = prev_grid[row_has_finite]
                keep_best_action_id = best_action_id[row_has_finite]
                keep_best_reward = best_reward[row_has_finite]

                self.value_curr[keep_soc_ids, keep_prev_ids] = keep_best_reward.astype(np.float32, copy=False)
                self.policy_reward_action_id[time_step, keep_soc_ids, keep_prev_ids] = keep_best_action_id.astype(
                    self.action_id_dtype,
                    copy=False,
                )

            self.value_curr, self.value_next = self.value_next, self.value_curr

        self.value_t0 = np.array(self.value_next, dtype=np.float32, copy=True)
        init_soc_id = int(self.DP_EMS_agent.soc_to_state_ids(np.array([self.env.state_init], dtype=np.float64))[0])
        if int(self.policy_reward_action_id[0, init_soc_id, initial_prev_fc_id]) == int(self.invalid_action_id):
            raise RuntimeError(
                "backward DP did not populate the initial 2D state; "
                "initial state is infeasible under the current discretization or terminal constraints"
            )
        self.save_backward_artifacts()

    def get_forward_policy(self):
        near_s0_list = []
        prev_fc_trace = []

        self.DP_EMS_agent.reset_forward_state(soc=self.env.state_init, prev_fc_power=0.0)
        travel = 0.0
        fcs_soh = 1.0
        bat_soh = 1.0

        soc = float(self.env.state_init)
        soc_idx, near_soc = self.find_soc_idx(soc)
        prev_fc_idx = int(self.DP_EMS_agent.initial_prev_fc_state_id())
        prev_fc_power = float(self.prev_fc_states[prev_fc_idx])

        info = {}
        for step in range(self.time_steps - 1):
            near_s0_list.append(near_soc)
            prev_fc_trace.append(prev_fc_power)

            action_id = int(self.policy_reward_action_id[step, soc_idx, prev_fc_idx])
            if action_id == int(self.invalid_action_id):
                raise RuntimeError(
                    f"invalid policy lookup at step={step}, soc_idx={soc_idx}, prev_fc_idx={prev_fc_idx}"
                )
            action = float(self.actions[action_id])

            self.policy_action_index[step] = action_id
            self.policy_action_value[step] = action
            self.policy_prev_fc_state[step] = prev_fc_idx
            self.policy_prev_fc_power[step] = prev_fc_power

            car_spd = float(self.env.speed_list[step])
            car_acc = float(self.env.acc_list[step])
            soc_new, info = self.execute(car_spd, car_acc, action, soc)

            soc_idx_new, near_soc_new = self.find_soc_idx(soc_new)
            soc = float(near_soc_new)
            self.DP_EMS_agent.SOC = soc
            info["SOC"] = soc

            prev_fc_idx_new, prev_fc_power_new = self.find_prev_fc_idx(self.DP_EMS_agent.FCHEV.P_FC_old)
            self.DP_EMS_agent.FCHEV.P_FC_old = prev_fc_power_new

            fcs_de = float(info.get("FCS_De", 0.0))
            batt_dsoh = float(info.get("dsoh", 0.0))
            travel += car_spd * self.DP_EMS_agent.FCHEV.simtime
            fcs_soh = max(0.0, fcs_soh - fcs_de)
            bat_soh = max(0.0, bat_soh - batt_dsoh)

            info.update({
                "travel": travel,
                "FCS_SOH": fcs_soh,
                "SOH": bat_soh,
                "speed": car_spd,
                "acc": car_acc,
                "prev_fc_state_id": prev_fc_idx,
                "current_fc_state_id": prev_fc_idx_new,
                "prev_fc_power": prev_fc_power,
                "current_fc_power": prev_fc_power_new,
            })
            for key in self.info_dict.keys():
                self.info_dict[key].append(info.get(key, 0.0))

            soc_idx = soc_idx_new
            near_soc = near_soc_new
            prev_fc_idx = prev_fc_idx_new
            prev_fc_power = prev_fc_power_new

        self.info_dict.update({
            "policy_action_value": self.policy_action_value.tolist(),
            "policy_action_index": self.policy_action_index.tolist(),
            "policy_prev_fc_power": self.policy_prev_fc_power.tolist(),
            "policy_prev_fc_state": self.policy_prev_fc_state.tolist(),
            "near_s0_list": near_s0_list,
            "prev_fc_trace": prev_fc_trace,
        })

        print("---dynamic programming finished!---")

        travel_km = float(info["travel"]) / 1000.0
        h2 = float(np.sum(self.info_dict["h2_fcs"]))
        h2_100 = h2 / travel_km * 100.0
        h2_batt = float(np.sum(self.info_dict["h2_batt"]))
        h2_batt_100 = h2_batt / travel_km * 100.0
        h2_equal = float(np.sum(self.info_dict["h2_equal"]))
        h2_equal_100 = h2_equal / travel_km * 100.0
        objective_cost = float(np.sum(self.info_dict["objective_cost"]))
        objective_100 = objective_cost / travel_km * 100.0
        self.info_dict.update({
            "h2_100": h2_100,
            "h2_batt_100": h2_batt_100,
            "h2_equal_100": h2_equal_100,
            "objective_100": objective_100,
        })

        soc_final = float(info["SOC"])
        fcs_soh_final = float(info["FCS_SOH"])
        bat_soh_final = float(info["SOH"])
        print(
            "\nDP-EMS-2D: travel %.3fkm, SOC %.4f, FCS-SOH %.6f, Bat-SOH %.6f"
            % (travel_km, soc_final, fcs_soh_final, bat_soh_final)
        )
        print(
            "DP-EMS-2D: H2_100km %.1fg, H2eq_100km %.1fg (batt %.1fg), objective_100km %.2f"
            % (h2_100, h2_equal_100, h2_batt_100, objective_100)
        )
        avg_h2_cost = float(np.mean(self.info_dict["h2_cost"]))
        avg_soc_cost = float(np.mean(self.info_dict["soc_cost"]))
        avg_fcs_soh_cost = float(np.mean(self.info_dict["fcs_soh_cost"]))
        avg_batt_soh_cost = float(np.mean(self.info_dict["batt_soh_cost"]))
        print(
            "DP-EMS-2D: avg cost h2eq %.4f, soc %.4f, fcs_soh %.4f, batt_soh %.4f"
            % (avg_h2_cost, avg_soc_cost, avg_fcs_soh_cost, avg_batt_soh_cost)
        )


if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    scenario = args.scenario_name
    dp_env = DP_Env(scenario)
    DP_EMS_Agent = DP_EMS_agent_new(
        w_soc=args.w_soc,
        soc0=args.soc0,
        SOC_MODE=args.MODE,
        eq_h2_batt_coef=args.eq_h2_batt_coef,
        soc_target=args.soc_target,
    )
    if int(args.dp_prev_fc_state_count) > 0:
        prev_fc_count = int(DP_EMS_Agent.build_uniform_prev_fc_grid(int(args.dp_prev_fc_state_count)).size)
    else:
        prev_fc_count = int(DP_EMS_Agent.build_prev_fc_grid(actions=dp_env.actions.astype(np.float64)).size)

    prev_fc_tag = int(args.dp_prev_fc_state_count) if int(args.dp_prev_fc_state_count) > 0 else prev_fc_count
    datadir = (
        "./DP_result/"
        + scenario
        + "_w%d" % args.w_soc
        + "_"
        + args.file_v
        + "_2d_soc_fcpower_pf%d_chunk%d" % (prev_fc_tag, int(args.dp_soc_chunk_size))
    )
    os.makedirs(datadir, exist_ok=True)

    print("scenario name: %s" % scenario)
    print(
        "\nstep %d * soc %d * prev_fc %d * action %d: %d"
        % (
            dp_env.time_steps,
            dp_env.states.shape[0],
            prev_fc_count,
            dp_env.actions.shape[0],
            dp_env.time_steps * dp_env.states.shape[0] * prev_fc_count * dp_env.actions.shape[0],
        )
    )
    print("2D DP config: prev_fc_states=%d, soc_chunk_size=%d" % (prev_fc_count, int(args.dp_soc_chunk_size)))
    DP_brain = DP_brain_2D(
        env=dp_env,
        DP_EMS_Agent=DP_EMS_Agent,
        output_dir=datadir,
        gamma=args.gamma,
        soc_chunk_size=args.dp_soc_chunk_size,
        prev_fc_state_count=args.dp_prev_fc_state_count,
    )

    DP_brain.DP_backward()

    end_calculate = time.time()
    calculation_time = end_calculate - start_time
    print("\ntime for backward calculation: %.2fs" % calculation_time)
    print("\nreward action policy saved in: %s" % DP_brain.policy_reward_action_path)

    DP_brain.get_forward_policy()
    scio.savemat(os.path.join(datadir, "DP_EMS_info.mat"), mdict={"DP_EMS_info": DP_brain.info_dict})
    print("\nsaved data in dir: %s" % os.path.join(datadir, "DP_EMS_info.mat"))

    end_time = time.time()
    spent_time = end_time - end_calculate
    print("\ntime for forward_policy: %.2fs" % spent_time)
