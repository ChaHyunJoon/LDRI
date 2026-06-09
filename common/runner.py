import datetime
import os
import numpy as np
import scipy.io as scio
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from common.memory import MemoryBuffer
from common.utils import summarize_fc_efficiency
from sac import SAC


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.buffer = MemoryBuffer(args)
        self.SAC_agent = SAC(args)
        # configuration
        self.episode_num = args.max_episodes
        self.episode_step = args.episode_steps
        self.start_episode = max(0, int(getattr(args, "start_episode", 0)))
        self.DONE = {}
        self.save_path = self.args.save_dir+'/'+self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path_episode = self.save_path+'/episode_data'
        if not os.path.exists(self.save_path_episode):
            os.makedirs(self.save_path_episode)
        # random seed (일관된 출력을 위한 유사난수 -> Reproducibility 확보)
        if self.args.random_seed:
            self.seed = np.random.randint(100)
        else:
            self.seed = 93
        # tensorboard
        fileinfo = args.scenario_name
        self.writer = SummaryWriter(args.log_dir+'/{}_{}_{}_seed{}'.format
                                    (datetime.datetime.now().strftime("%m-%d_%H-%M"),
                                     fileinfo, self.args.DRL, self.seed))
        
    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        print("Random seeds have been set to %d !\n"%self.seed)
    
    def run_SAC(self):
        reward = []  # reward of each episode
        c_loss_1 = []
        c_loss_2 = []
        a_loss = []
        en_loss = []
        h2_100_list = []
        h2_batt_100_list = []
        h2_equal_100_list = []
        objective_100_list = []  # objective cost per 100 km
        FCS_SoH = []
        Batt_SoH = []
        SOC = []
        lr_recorder = {'lrcr': [], 'lrac': [], 'lral': []}
        updates = 0  # for tensorboard counter
        best_reward = -float("inf")
        save_best_enabled = getattr(self.args, "save_best", True)
        episode = self.start_episode
        episodes_done = 0
        with tqdm(total=self.episode_num) as pbar:
            while episodes_done < self.episode_num:
                state = self.env.reset()  # reset the environment
                episode_steps = self.episode_step
                episode_reward = []
                c_loss_1_one_ep = []
                c_loss_2_one_ep = []
                a_loss_one_ep = []
                en_loss_one_ep = []
                alpha_value_ep = []
                info = []
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
                                'soc_in_bounds': []}

                for episode_step in range(episode_steps):
                    with torch.no_grad():
                        action = self.SAC_agent.select_action(state, evaluate=False)
                    state_next, step_reward, done, info = self.env.step(action, episode_step)
                    # when done is True: unsafe, stop
                    self.buffer.store(state, action, step_reward, state_next, done)
                    state = state_next
                    # save data
                    for key in episode_info.keys():
                        episode_info[key].append(info.get(key, np.nan))
                    episode_reward.append(step_reward)
                    # learn
                    if self.buffer.currentSize >= 10*self.args.batch_size:
                        transition = self.buffer.random_sample()
                        critic_loss_1, critic_loss_2, actor_loss, alpha_loss, alpha = self.SAC_agent.learn(transition)
                        # save to tensorboard
                        self.writer.add_scalar('loss/critic_1', critic_loss_1, updates)
                        self.writer.add_scalar('loss/critic_2', critic_loss_2, updates)
                        self.writer.add_scalar('loss/actor', actor_loss, updates)
                        self.writer.add_scalar('loss/alpha_loss', alpha_loss, updates)
                        self.writer.add_scalar('entropy/alpha_value', alpha, updates)
                        self.writer.add_scalar('reward/step_reward', step_reward, updates)
                        updates += 1
                        # save in .mat
                        c_loss_1_one_ep.append(critic_loss_1)
                        c_loss_2_one_ep.append(critic_loss_2)
                        a_loss_one_ep.append(actor_loss)
                        en_loss_one_ep.append(alpha_loss)
                        alpha_value_ep.append(alpha)
                    if done and episode not in self.DONE.keys():
                        print('\nfailure in step %d of episode %d'%(episode_step, episode))
                        self.DONE.update({episode: episode_step})
                    '''
                    if done:
                        # save data in .mat on early termination
                        episode_info.update({'alpha': alpha_value_ep})
                        self.SAC_agent.save_net(episode)
                        datadir = self.save_path_episode+'/data_ep%d.mat'%episode
                        scio.savemat(datadir, mdict=episode_info)
                        break
                    '''
                    # save data in .mat
                    # if episode_step+1 == self.episode_step:
                    if episode_step+1 == episode_steps:
                        # only save the last 10 episode
                        # save alpha value of each time step in each episode
                        episode_info.update({'alpha': alpha_value_ep})
                        # save network parameters
                        self.SAC_agent.save_net(episode)
                        # save all data in one episode info
                        datadir = self.save_path_episode+'/data_ep%d.mat'%episode
                        scio.savemat(datadir, mdict=episode_info)
                # record current optimizer learning rates
                lrcr, lrac, lral = self.SAC_agent.get_lrs()
                lr_recorder['lrcr'].append(lrcr)
                lr_recorder['lrac'].append(lrac)
                lr_recorder['lral'].append(lral)
                # show episode data
                travel = info['travel']/1000  # km
                h2 = sum(episode_info['h2_fcs'])  # g
                h2_100 = h2/travel*100
                h2_100_list.append(h2_100)
                h2_batt = sum(episode_info['h2_batt'])  # g
                h2_batt_100 = h2_batt/travel*100
                h2_batt_100_list.append(h2_batt_100)
                h2_equal = sum(episode_info['h2_equal'])  # g
                h2_equal_100 = h2_equal/travel*100
                h2_equal_100_list.append(h2_equal_100)
                objective_cost = sum(episode_info['objective_cost'])
                objective_100 = objective_cost/travel*100
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
                # print
                soc = info['SOC']
                fcs_soh = info['FCS_SOH']
                bat_soh = info['SOH']
                print('\nepi %d: travel %.3fkm, SOC %.4f, FCS-SOH %.6f, Bat-SOH %.6f'
                      % (episode, travel, soc, fcs_soh, bat_soh))
                print('epi %d: H2_100km %.1fg, H2eq_100km %.1fg (batt %.1fg), objective_100km %.2f'%
                      (episode, h2_100, h2_equal_100, h2_batt_100, objective_100))
                print('epi %d: SOC[min/mean/max]=%.3f/%.3f/%.3f, in_bounds_rate=%.1f%%, len=%d'
                      % (episode, soc_min, soc_mean, soc_max, soc_in_bounds_rate, ep_len))
                print('epi %d: mean_cost soc %.3f, h2eq %.3f, fcs %.3f, batt %.3f, obj %.3f'
                      % (episode, mean_soc_cost, mean_h2_cost, mean_fcs_cost, mean_batt_cost,
                         mean_objective))
                print('epi %d: P_fc min/mean/max %.2f/%.2f/%.2f kW, P_batt_req min/mean/max %.2f/%.2f/%.2f kW, fce_eff mean %.3f, fc_on %.1f%%'
                      % (episode, min_p_fc, mean_p_fc, max_p_fc, min_p_batt, mean_p_batt, max_p_batt, mean_fce_eff, pct_fc_on))
                # cumulative reward over the episode and average losses
                ep_r = float(np.sum(episode_reward)) if episode_reward else 0.0
                ep_c1 = np.mean(c_loss_1_one_ep)
                ep_c2 = np.mean(c_loss_2_one_ep)
                ep_a = np.mean(a_loss_one_ep)
                ep_en = np.mean(en_loss_one_ep)
                if alpha_value_ep:
                    ep_alpha = float(np.mean(alpha_value_ep))
                else:
                    ep_alpha = float(self.SAC_agent.alpha) if self.SAC_agent is not None else 0.0
                print('epi %d: ep_cumulative_r %.3f, c-loss1 %.4f, c-loss2 %.4f, a-loss %.4f, en-loss %.4f, alpha %.6f'
                      % (episode, ep_r, ep_c1, ep_c2, ep_a, ep_en, ep_alpha))
                print('epi %d: lr_critic %.6f, lr_actor %.6f, lr_alpha %.6f' % (episode, lrcr, lrac, lral))
                self.writer.add_scalar('episode/length', ep_len, episode)
                self.writer.add_scalar('episode/soc_min', soc_min, episode)
                self.writer.add_scalar('episode/soc_mean', soc_mean, episode)
                self.writer.add_scalar('episode/soc_max', soc_max, episode)
                self.writer.add_scalar('episode/soc_in_bounds_rate', soc_in_bounds_rate, episode)
                self.writer.add_scalar('episode/mean_soc_cost', mean_soc_cost, episode)
                self.writer.add_scalar('episode/mean_h2_cost', mean_h2_cost, episode)
                self.writer.add_scalar('episode/mean_fcs_soh_cost', mean_fcs_cost, episode)
                self.writer.add_scalar('episode/mean_batt_soh_cost', mean_batt_cost, episode)
                self.writer.add_scalar('episode/mean_objective_cost', mean_objective, episode)
                self.writer.add_scalar('episode/h2_100km', h2_100, episode)
                self.writer.add_scalar('episode/h2_batt_100km', h2_batt_100, episode)
                self.writer.add_scalar('episode/h2_equal_100km', h2_equal_100, episode)
                self.writer.add_scalar('episode/mean_alpha', ep_alpha, episode)
                self.writer.add_scalar('episode/min_p_fc', min_p_fc, episode)
                self.writer.add_scalar('episode/mean_p_fc', mean_p_fc, episode)
                self.writer.add_scalar('episode/max_p_fc', max_p_fc, episode)
                self.writer.add_scalar('episode/min_p_batt', min_p_batt, episode)
                self.writer.add_scalar('episode/mean_p_batt', mean_p_batt, episode)
                self.writer.add_scalar('episode/max_p_batt', max_p_batt, episode)
                self.writer.add_scalar('episode/mean_fce_eff', mean_fce_eff, episode)
                self.writer.add_scalar('episode/pct_fc_on', pct_fc_on, episode)
                # save best checkpoint
                if ep_r > best_reward:
                    best_reward = ep_r
                    if save_best_enabled:
                        self.SAC_agent.save_best(episode, ep_r)
                        print(f'best checkpoint updated: ep {episode}, reward {ep_r:.3f}')
                if self.SAC_agent is not None:
                    self.SAC_agent.step_lr_schedulers()
                reward.append(ep_r)
                c_loss_1.append(ep_c1)
                c_loss_2.append(ep_c2)
                a_loss.append(ep_a)
                en_loss.append(ep_en)
                FCS_SoH.append(fcs_soh)
                Batt_SoH.append(bat_soh)
                SOC.append(soc)
                episodes_done += 1
                pbar.update(1)
                episode += 1
        
        scio.savemat(self.save_path+'/reward.mat', mdict={'reward': reward})
        scio.savemat(self.save_path+'/critic_loss.mat', mdict={'c_loss_1': c_loss_1, 'c_loss_2': c_loss_2})
        scio.savemat(self.save_path+'/actor_loss.mat', mdict={'a_loss': a_loss})
        scio.savemat(self.save_path+'/entropy_loss.mat', mdict={'en_loss': en_loss})
        scio.savemat(self.save_path+'/lr_recorder.mat', mdict=lr_recorder)
        scio.savemat(self.save_path+'/h2.mat', mdict={'h2': h2_100_list})
        scio.savemat(self.save_path+'/h2_batt.mat', mdict={'h2_batt': h2_batt_100_list})
        scio.savemat(self.save_path+'/h2_equal.mat', mdict={'h2_equal': h2_equal_100_list})
        scio.savemat(self.save_path+'/objective.mat', mdict={'objective_100': objective_100_list})
        scio.savemat(self.save_path+'/FCS_SOH.mat', mdict={'FCS_SOH': FCS_SoH})
        scio.savemat(self.save_path+'/Batt_SOH.mat', mdict={'Batt_SOH': Batt_SoH})
        scio.savemat(self.save_path+'/SOC.mat', mdict={'SOC': SOC})

    def memory_info(self):
        print('\nbuffer counter:', self.buffer.counter)
        print('buffer current size:', self.buffer.currentSize)
        print('replay ratio: %.3f'%(self.buffer.counter/self.buffer.currentSize))
        print('failure:', self.DONE)
