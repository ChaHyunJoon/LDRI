from tqdm import tqdm
import os
import torch
import numpy as np
import time
import scipy.io as scio
import sys
from sac import SAC
from common.utils import summarize_fc_efficiency

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRAMS_PER_KILOGRAM = 1000.0
SECONDS_PER_HOUR = 3600.0
KWH_PER_US_GALLON_GASOLINE_EQ = 33.7
US_GALLON_TO_CUBIC_METER = 0.00378541
METER_TO_MILE = 0.000621371


def _as_1d_float(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.where(np.isfinite(arr), arr, 0.0)


def compute_fcev_reference_mpg(travel_m, h2_fcs_gps, p_batt_kw, dt_s=1.0):
    """FCEV Reference-style US MPG from H2 flow and battery power.

    This mirrors the Simulink block algebra:
    1. Convert H2 mass flow to gasoline-gallon-equivalent flow.
    2. Convert battery power to gasoline-gallon-equivalent flow using 33.7 kWh/US gal.
    3. Sum the flows, convert through m^3 if desired, integrate, then divide miles by US gallons.
    """
    travel_m = float(travel_m)
    if not np.isfinite(travel_m) or travel_m <= 1e-12:
        return np.nan, np.nan

    h2_gps = _as_1d_float(h2_fcs_gps)
    p_batt_kw = _as_1d_float(p_batt_kw)
    n = max(h2_gps.size, p_batt_kw.size)
    if n == 0:
        return np.nan, np.nan

    if h2_gps.size == 0:
        h2_gps = np.zeros(n, dtype=np.float64)
    elif h2_gps.size < n:
        h2_gps = np.pad(h2_gps, (0, n - h2_gps.size))

    if p_batt_kw.size == 0:
        p_batt_kw = np.zeros(n, dtype=np.float64)
    elif p_batt_kw.size < n:
        p_batt_kw = np.pad(p_batt_kw, (0, n - p_batt_kw.size))

    h2_gal_s = h2_gps / GRAMS_PER_KILOGRAM
    batt_gal_s = p_batt_kw / KWH_PER_US_GALLON_GASOLINE_EQ / SECONDS_PER_HOUR
    fuel_eq_gal_s = h2_gal_s + batt_gal_s
    fuel_eq_m3_s = fuel_eq_gal_s * US_GALLON_TO_CUBIC_METER
    total_fuel_eq_m3 = float(np.sum(fuel_eq_m3_s) * float(dt_s))
    total_fuel_eq_gal = total_fuel_eq_m3 / US_GALLON_TO_CUBIC_METER
    distance_mile = travel_m * METER_TO_MILE

    if not np.isfinite(total_fuel_eq_gal) or total_fuel_eq_gal <= 1e-12:
        return np.nan, total_fuel_eq_gal
    return distance_mile / total_fuel_eq_gal, total_fuel_eq_gal

class Evaluator:
    def __init__(self, args, env):
        self.args = args
        self.eva_episode = args.evaluate_episode
        self.episode_step = args.episode_steps
        self.env = env
        self.DRL_agent = SAC(args)
    
        eval_name = getattr(self.args, 'eval_name', self.args.scenario_name)
        self.save_path = self.args.eva_dir+'/'+eval_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path_episode = self.save_path+'/episode_data'
        if not os.path.exists(self.save_path_episode):
            os.makedirs(self.save_path_episode)
 
    def evaluate(self):
        rewards = []  # cumulative reward of each episode
        h2_100_list = []
        h2_batt_100_list = []
        h2_equal_100_list = []
        mpg_list = []
        fuel_eq_gal_list = []
        objective_100_list = []  # objective cost per 100 km
        FCS_SoH = []
        Batt_SoH = []
        SOC = []
        eval_stochastic = bool(getattr(self.args, "eval_stochastic", False))
        simtime_s = float(getattr(self.env.agent.FCHEV, "simtime", 1.0))
        for episode in tqdm(range(self.eva_episode)):
            state = self.env.reset()  # reset the environment
            episode_steps = self.episode_step
            episode_reward = []
            # data being saved in .mat
            episode_info = {'T_mot': [], 'W_mot': [], 'mot_eff': [], 'P_mot': [],
                            'P_fc': [], 'P_fce': [], 'fce_eff': [], 'FCS_SOH': [],
                            'P_dcdc': [], 'dcdc_eff': [], 'FCS_De': [], 'travel': [],
                            'd_s_s': [], 'd_low': [], 'd_high': [], 'd_l_c': [],
                            'EMS_reward': [], 'soc_cost': [], 'h2_fcs': [], 'h2_batt': [], 'h2_equal': [],
                            'objective_cost': [], 'h2_cost': [],
                            'batt_soh_cost': [], 'fcs_soh_cost': [],
                            'SOC': [], 'SOH': [], 'I': [], 'I_c': [],
                            'pack_OCV': [], 'pack_Vt': [], 'pack_power_out': [],
                            'P_batt_req': [], 'tep_a': [], 'dsoh': [],
                            'soc_in_bounds': [], }
            start_time = time.time()
            for episode_step in range(episode_steps):
                with torch.no_grad():
                    raw_action = self.DRL_agent.select_action(state, evaluate=True)
                action = raw_action
                state_next, step_reward, done, info = self.env.step(action, episode_step)
                state = state_next
                # save data
                for key in episode_info.keys():
                    episode_info[key].append(info.get(key, np.nan))
                episode_reward.append(step_reward)
                # save data in .mat
                if episode_step+1 == episode_steps:
                    datadir = self.save_path_episode+'/data_ep%d.mat'%episode
                    scio.savemat(datadir, mdict=episode_info)
                    # print
                    travel_m = float(info['travel'])
                    f_travel = travel_m / 1000.0
                    h2 = sum(episode_info['h2_fcs'])  # g
                    h2_100 = h2/f_travel*100
                    h2_100_list.append(h2_100)
                    mpg, fuel_eq_total_gal = compute_fcev_reference_mpg(
                        travel_m=travel_m,
                        h2_fcs_gps=episode_info['h2_fcs'],
                        p_batt_kw=episode_info['P_batt_req'],
                        dt_s=simtime_s,
                    )
                    mpg_list.append(mpg)
                    fuel_eq_gal_list.append(fuel_eq_total_gal)
                    h2_batt = sum(episode_info['h2_batt'])  # g
                    h2_batt_100 = h2_batt/f_travel*100
                    h2_batt_100_list.append(h2_batt_100)
                    h2_equal = sum(episode_info['h2_equal'])  # g
                    h2_equal_100 = h2_equal/f_travel*100
                    h2_equal_100_list.append(h2_equal_100)
                    objective_cost = sum(episode_info['objective_cost'])
                    objective_100 = objective_cost/f_travel*100
                    objective_100_list.append(objective_100)
                    soc_arr = np.asarray(episode_info['SOC'], dtype=np.float64)
                    soc_min = float(np.min(soc_arr))
                    soc_max = float(np.max(soc_arr))
                    soc_mean = float(np.mean(soc_arr))
                    soc_in_bounds = np.asarray(episode_info['soc_in_bounds'], dtype=np.float64)
                    soc_in_bounds_rate = float(np.mean(soc_in_bounds)) * 100.0
                    mean_soc_cost = float(np.mean(episode_info['soc_cost']))
                    mean_h2_cost = float(np.mean(episode_info['h2_cost']))
                    mean_fcs_cost = float(np.mean(episode_info['fcs_soh_cost']))
                    mean_batt_cost = float(np.mean(episode_info['batt_soh_cost']))
                    mean_objective = float(np.mean(episode_info['objective_cost']))
                    p_fc = np.asarray(episode_info['P_fc'], dtype=np.float64)
                    p_batt = np.asarray(episode_info['P_batt_req'], dtype=np.float64)
                    min_p_fc = float(np.min(p_fc))
                    min_p_batt = float(np.min(p_batt))
                    mean_p_fc = float(np.mean(p_fc))
                    mean_p_batt = float(np.mean(p_batt))
                    max_p_fc = float(np.max(p_fc))
                    max_p_batt = float(np.max(p_batt))
                    mean_fce_eff, pct_fc_on = summarize_fc_efficiency(
                        p_fc,
                        episode_info['fce_eff'],
                        on_threshold=self.env.agent.FCHEV.P_FC_off,
                    )
                    ep_len = len(episode_reward)
                    soc = info['SOC']
                    bat_soh = info['SOH']
                    fcs_soh = info['FCS_SOH']
                    SOC.append(soc)
                    Batt_SoH.append(bat_soh)
                    FCS_SoH.append(fcs_soh)
                    print('\nepi %d: travel %.3fkm, SOC %.4f, Bat-SOH %.6f, FCS-SOH %.6f'
                          %(episode, f_travel, soc, bat_soh, fcs_soh))
                    print('epi %d: H2_100km %.1fg, H2eq_100km %.1fg (batt %.1fg), MPG %.2f, fuel_eq %.4f USgal, objective_100km %.2f'%
                          (episode, h2_100, h2_equal_100, h2_batt_100, mpg, fuel_eq_total_gal, objective_100))
                    print('epi %d: SOC[min/mean/max]=%.3f/%.3f/%.3f, in_bounds_rate=%.1f%%, len=%d'
                          % (episode, soc_min, soc_mean, soc_max, soc_in_bounds_rate, ep_len))
                    print('epi %d: mean_cost soc %.3f, h2eq %.3f, fcs %.3f, batt %.3f, obj %.3f'
                          % (episode, mean_soc_cost, mean_h2_cost, mean_fcs_cost, mean_batt_cost,
                             mean_objective))
                    print('epi %d: P_fc min/mean/max %.2f/%.2f/%.2f kW, P_batt_req min/mean/max %.2f/%.2f/%.2f kW, fce_eff mean %.3f, fc_on %.1f%%'
                          % (episode, min_p_fc, mean_p_fc, max_p_fc, min_p_batt, mean_p_batt, max_p_batt, mean_fce_eff, pct_fc_on))
                
            end_time = time.time()
            spent_time = end_time-start_time
            # save reward
            ep_r = float(np.sum(episode_reward)) if episode_reward else 0.0
            rewards.append(ep_r)
            # print
            print('episode %d: reward %.3f, time spent: %.3fs'
                  %(episode, ep_r, spent_time))
    
        scio.savemat(self.save_path+'/reward.mat', mdict={'reward': rewards})
        scio.savemat(self.save_path+'/h2.mat', mdict={'h2': h2_100_list})
        scio.savemat(self.save_path+'/h2_batt.mat', mdict={'h2_batt': h2_batt_100_list})
        scio.savemat(self.save_path+'/h2_equal.mat', mdict={'h2_equal': h2_equal_100_list})
        scio.savemat(self.save_path+'/mpg.mat', mdict={'mpg': mpg_list})
        scio.savemat(self.save_path+'/fuel_eq_gal.mat', mdict={'fuel_eq_gal': fuel_eq_gal_list})
        scio.savemat(self.save_path+'/objective.mat', mdict={'objective_100': objective_100_list})
        scio.savemat(self.save_path+'/FCS_SOH.mat', mdict={'FCS_SOH': FCS_SoH})
        scio.savemat(self.save_path+'/Batt_SOH.mat', mdict={'Batt_SoH': Batt_SoH})
        scio.savemat(self.save_path+'/SOC.mat', mdict={'SOC': SOC})
    
