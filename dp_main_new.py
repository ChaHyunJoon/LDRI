"""
Dynamic Programming, energy management
"""
import os
import time
import numpy as np
from tqdm import tqdm
import scipy.io as scio
from common.arguments import get_args
from DP_env import DP_Env
from DP_EMS_agent import DP_EMS_agent

class DP_brain:
    def __init__(self, env, DP_EMS_Agent):
        self.DP_EMS_agent = DP_EMS_Agent
        self.env = env
        self.gamma = 0.99
        self.states = self.env.states
        self.actions = self.env.actions
        # Forward Simulation을 위한 결과 저장용 배열
        self.policy_as_action_id = np.full(self.env.time_steps, -1.0, dtype=np.float32)
        self.policy_as_reward_id = np.full(self.env.time_steps, -1.0, dtype=np.float32)
        '''
        DP 테이블 (크기: 상태 수 x 시간 단계 수)
        optimal_action_id_table: 각 상태/시간에서 최적의 행동(Action ID)을 저장
        optimal_reward_table: 각 상태/시간에서 얻을 수 있는 최대 누적 보상(Value Function, V)
        '''
        self.optimal_action_id_table = np.zeros((len(self.states), self.env.time_steps), dtype=np.int32)
        self.optimal_reward_table = np.zeros((len(self.states), self.env.time_steps), dtype=np.float32)
        self.optimal_reward_id_table = np.zeros(self.optimal_reward_table.shape, dtype=np.int32)
        self.optimal_action_table = np.full(self.optimal_reward_table.shape, -1.0, dtype=np.float32)
        self.optimal_reward_action_table = np.full(self.optimal_reward_table.shape, -1.0, dtype=np.float32)
        # self.deltas = np.zeros(self.optimal_action_table.shape, dtype=np.float32)
        self.info_dict = {'T_mot': [], 'W_mot': [], 'mot_eff': [], 'P_mot': [],
                          'P_fc': [], 'P_fce': [], 'fce_eff': [], 'FCS_SOH': [],
                          'P_dcdc': [], 'dcdc_eff': [], 'FCS_De': [], 'travel': [],
                          'd_s_s': [], 'd_low': [], 'd_high': [], 'd_l_c': [],
                          'speed': [], 'acc': [],
                          'EMS_reward': [], 'soc_cost': [], 'h2_fcs': [], 'h2_batt': [], 'h2_equal': [],
                          'objective_cost': [], 'h2_cost': [], 'batt_soh_cost': [], 'fcs_soh_cost': [],
                          'SOC': [], 'SOH': [], 'I': [], 'I_c': [],
                          'pack_OCV': [], 'pack_Vt': [],
                          'pack_power_out': [], 'P_batt_req': [], 'tep_a': [], 'dsoh': []}
    
    def DP_backward(self):
        terminal_step = self.env.time_steps - 1
        # --- 1. Setting Terminal Cost ---
        soc_target = 0.6
        self.DP_EMS_agent.SOC_target = soc_target
        state_min = float(self.states[0])
        state_inc = float(self.env.state_increment)
        target_id = int(np.rint((soc_target - state_min) / state_inc))
        self.optimal_reward_table[:, terminal_step] = -np.inf
        if 0 <= target_id < len(self.states):
            self.optimal_reward_table[target_id, terminal_step] = 0.0
        # --- 2. Forward Reachability Window ---
        soc_init = self.env.state_init
        max_delta_soc = 0.01  # Approx max SOC change per step for FCHEV.
        min_soc = float(np.min(self.states))
        max_soc = float(np.max(self.states))
        min_reachable_soc = np.zeros(self.env.time_steps, dtype=np.float32)
        max_reachable_soc = np.zeros(self.env.time_steps, dtype=np.float32)
        min_reachable_soc[0] = soc_init
        max_reachable_soc[0] = soc_init
        for t in range(1, self.env.time_steps):
            min_reachable_soc[t] = min_reachable_soc[t - 1] - max_delta_soc
            max_reachable_soc[t] = max_reachable_soc[t - 1] + max_delta_soc
        min_reachable_soc = np.clip(min_reachable_soc, min_soc, max_soc)
        max_reachable_soc = np.clip(max_reachable_soc, min_soc, max_soc)
        # --- 2b. Backward Reachability Window (to terminal target) ---
        min_to_target_soc = np.zeros(self.env.time_steps, dtype=np.float32)
        max_to_target_soc = np.zeros(self.env.time_steps, dtype=np.float32)
        for t in range(self.env.time_steps):
            remaining_steps = terminal_step - t
            min_to_target_soc[t] = soc_target - remaining_steps * max_delta_soc
            max_to_target_soc[t] = soc_target + remaining_steps * max_delta_soc
        min_to_target_soc = np.clip(min_to_target_soc, min_soc, max_soc)
        max_to_target_soc = np.clip(max_to_target_soc, min_soc, max_soc)

        # --- 2c. Backward fast cache ---
        actions_eval = self.actions.astype(np.float64)
        self.DP_EMS_agent.prepare_backward_cache(
            actions=actions_eval,
            speed_list=self.env.speed_list,
            acc_list=self.env.acc_list,
            states=self.states,
        )

        # --- 3. Backward Time Loop ---
        max_state_id = len(self.states) - 1
        for time_step in tqdm(range(self.env.time_steps-2, -1, -1)):
            # Default column values for infeasible states
            self.optimal_reward_table[:, time_step] = -np.inf
            self.optimal_reward_id_table[:, time_step] = 0
            self.optimal_action_id_table[:, time_step] = 0
            self.optimal_reward_action_table[:, time_step] = -1.0
            self.optimal_action_table[:, time_step] = -1.0

            valid_mask = (
                (self.states >= min_reachable_soc[time_step])
                & (self.states <= max_reachable_soc[time_step])
                & (self.states >= min_to_target_soc[time_step])
                & (self.states <= max_to_target_soc[time_step])
            )
            valid_ids = np.flatnonzero(valid_mask)
            if valid_ids.size == 0:
                continue

            # State transition and reward for all valid states and all actions at once.
            soc_next, DP_EMS_reward = self.DP_EMS_agent.batch_transition_reward(time_step, valid_ids)
            objective_cost = -DP_EMS_reward

            # Next-state value lookup (advanced indexing: [state, action] -> reward).
            next_state_id = np.rint((soc_next - state_min) / state_inc).astype(np.int32)
            next_state_id = np.clip(next_state_id, 0, max_state_id)
            next_reward = self.optimal_reward_table[next_state_id, time_step + 1]

            # Bellman update in matrix form.
            feasible = np.isfinite(next_reward)
            total_reward = np.where(feasible, DP_EMS_reward + self.gamma * next_reward, -np.inf)
            total_cost = np.where(feasible, objective_cost - self.gamma * next_reward, np.inf)

            row_has_finite = np.any(np.isfinite(total_reward), axis=1)
            if not np.any(row_has_finite):
                continue

            row_idx = np.arange(valid_ids.size)
            best_reward_action_id = np.argmax(total_reward, axis=1)
            best_cost_action_id = np.argmin(total_cost, axis=1)
            best_reward = total_reward[row_idx, best_reward_action_id]

            keep_ids = valid_ids[row_has_finite]
            keep_reward_action = best_reward_action_id[row_has_finite]
            keep_cost_action = best_cost_action_id[row_has_finite]

            # Store optimal values in DP tables.
            self.optimal_reward_table[keep_ids, time_step] = best_reward[row_has_finite]
            self.optimal_reward_id_table[keep_ids, time_step] = keep_reward_action
            self.optimal_action_id_table[keep_ids, time_step] = keep_cost_action
            self.optimal_reward_action_table[keep_ids, time_step] = actions_eval[keep_reward_action]
            self.optimal_action_table[keep_ids, time_step] = actions_eval[keep_cost_action]
         
    def execute(self, car_spd, car_acc, action, soc):
        soc_new = self.DP_EMS_agent.execute(
            action=action, car_spd=car_spd, car_acc=car_acc, soc=soc
        )
        self.DP_EMS_agent.get_reward()
        info = self.DP_EMS_agent.get_info()
        return soc_new, info
    
    def find_s_idx(self, SOC_new):
        delta_list = abs(SOC_new-self.env.states)
        near_s_id = np.argmin(delta_list)
        near_s = self.states[near_s_id]
        return near_s_id, near_s
 
    def get_forward_policy(self):
        near_s0_list = []
        self.DP_EMS_agent.SOC = self.env.state_init
        self.DP_EMS_agent.h2_fcs = 0.0
        self.DP_EMS_agent.h2_batt = 0.0
        self.DP_EMS_agent.h2_equal = 0.0
        self.DP_EMS_agent.dSOH_FCS = 0.0
        self.DP_EMS_agent.P_batt = 0.0
        self.DP_EMS_agent.dSOH_batt = 0.0
        self.DP_EMS_agent.info = {}
        self.DP_EMS_agent.FCHEV.P_FC_old = 0.0
        travel = 0.0
        fcs_soh = 1.0
        bat_soh = 1.0
        s0 = self.env.state_init
        s0_idx, near_s0 = self.find_s_idx(s0)
        info = {}
        # Backward DP uses terminal index as value-only state (no control at terminal step).
        for step in range(self.env.time_steps - 1):
            near_s0_list.append(near_s0)
            car_spd = self.env.speed_list[step]
            car_acc = self.env.acc_list[step]
            # get action
            act1 = float(self.optimal_action_table[s0_idx, step])
            self.policy_as_action_id[step] = act1
            act2 = float(self.optimal_reward_action_table[s0_idx, step])
            self.policy_as_reward_id[step] = act2
            act = act2
            
            s_new, info = self.execute(car_spd, car_acc, act, s0)
            s_idx, near_s = self.find_s_idx(s_new)  # id of new state
            # Snap to DP grid to keep rollout consistent with discrete DP policy.
            s_new = float(near_s)
            self.DP_EMS_agent.SOC = s_new
            info['SOC'] = s_new
            fcs_de = info['FCS_De']
            batt_dsoh = info['dsoh']
            travel += car_spd * self.DP_EMS_agent.FCHEV.simtime
            fcs_soh = max(0.0, fcs_soh - fcs_de)
            bat_soh = max(0.0, bat_soh - batt_dsoh)
            info.update({
                'travel': travel,
                'FCS_SOH': fcs_soh,
                'SOH': bat_soh,
                'speed': car_spd,
                'acc': car_acc,
            })
            for key in self.info_dict.keys():
                self.info_dict[key].append(info.get(key, 0.0))
            
            near_s0 = near_s
            s0_idx = s_idx
            s0 = s_new
            
        self.info_dict.update({'policy_as_action': self.policy_as_action_id.tolist(),
                               'policy_as_reward': self.policy_as_reward_id.tolist(),
                               'near_s0_list': near_s0_list})
        print("---dynamic programming finished!---")
        # show data
        travel = info['travel']/1000  # km
        h2 = sum(self.info_dict['h2_fcs'])  # g
        h2_100 = h2/travel*100
        h2_batt = sum(self.info_dict['h2_batt'])  # g
        h2_batt_100 = h2_batt/travel*100
        h2_equal = sum(self.info_dict['h2_equal'])  # g
        h2_equal_100 = h2_equal/travel*100
        objective_cost = sum(self.info_dict['objective_cost'])
        objective_100 = objective_cost/travel*100
        self.info_dict.update({'h2_100': h2_100, 'h2_batt_100': h2_batt_100,
                               'h2_equal_100': h2_equal_100, 'objective_100': objective_100})
        # print
        soc = info['SOC']
        fcs_soh = info['FCS_SOH']
        bat_soh = info['SOH']
        print('\nDP-EMS: travel %.3fkm, SOC %.4f, FCS-SOH %.6f, Bat-SOH %.6f'
              %(travel, soc, fcs_soh, bat_soh))
        print('DP-EMS: H2_100km %.1fg, H2eq_100km %.1fg (batt %.1fg), objective_100km %.2f'
              %(h2_100, h2_equal_100, h2_batt_100, objective_100))
        avg_h2_cost = float(np.mean(self.info_dict['h2_cost']))
        avg_soc_cost = float(np.mean(self.info_dict['soc_cost']))
        avg_fcs_soh_cost = float(np.mean(self.info_dict['fcs_soh_cost']))
        avg_batt_soh_cost = float(np.mean(self.info_dict['batt_soh_cost']))
        print('DP-EMS: avg cost h2eq %.4f, soc %.4f, fcs_soh %.4f, batt_soh %.4f'
              %(avg_h2_cost, avg_soc_cost, avg_fcs_soh_cost, avg_batt_soh_cost))
 
 
if __name__ == "__main__":
    strat_tiem = time.time()
    args = get_args()
    scenario = args.scenario_name  # CTUDC, WVU, JN
    dp_env = DP_Env(scenario)
    print('scenario name: %s'%scenario)
    print('\nstep %d * state %d * action %d: %d'%
          (dp_env.time_steps, dp_env.states.shape[0], dp_env.actions.shape[0],
           dp_env.states.shape[0]*dp_env.actions.shape[0]*dp_env.time_steps))
    DP_EMS_Agent = DP_EMS_agent(
        w_soc=args.w_soc,
        soc0=args.soc0,
        SOC_MODE=args.MODE,
        eq_h2_batt_coef=args.eq_h2_batt_coef,
        soc_target=args.soc_target,
    )
    DP_brain = DP_brain(dp_env, DP_EMS_Agent)
    DP_brain.DP_backward()
    # save data dir
    datadir = './DP_result/' + scenario + '_w%d' % args.w_soc + '_'+args.file_v
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    # save backward data
    dataname_back1 = '/action_id.mat'
    scio.savemat(datadir+dataname_back1, mdict={'action_id': DP_brain.optimal_action_id_table})
    dataname_back2 = '/reward_id.mat'
    scio.savemat(datadir+dataname_back2, mdict={'reward_id': DP_brain.optimal_reward_id_table})
    dataname_back3 = '/optimal_reward.mat'
    scio.savemat(datadir+dataname_back3, mdict={'optimal_reward': DP_brain.optimal_reward_table})
    end_claculate = time.time()
    calculation_time = end_claculate-strat_tiem
    print("\ntime for calculation: %.2fs"%calculation_time)
    # forward
    DP_brain.get_forward_policy()
    dataname = '/DP_EMS_info.mat'
    scio.savemat(datadir+dataname, mdict={'DP_EMS_info': DP_brain.info_dict})
    print("\nsaved data in dir: %s"%(datadir+dataname))

    end_time = time.time()
    spent_time = end_time-end_claculate
    print("\ntime for forward_policy: %.2fs"%spent_time)
    
