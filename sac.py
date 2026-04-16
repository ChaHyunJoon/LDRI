import os
import json
import math
import re
import torch
import torch.nn.functional as F
import torch.optim as opt
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from common.network import QNetwork, GaussianPolicy

class SAC:
    def __init__(self, args):
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.automatic_entropy_tuning = args.auto_tune
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.actor_lr_min = float(getattr(args, "lr_actor_min", args.lr_actor))
        self.critic_lr_min = float(getattr(args, "lr_critic_min", args.lr_critic))
        if self.actor_lr_min > float(args.lr_actor):
            raise ValueError("lr_actor_min must be <= lr_actor")
        if self.critic_lr_min > float(args.lr_critic):
            raise ValueError("lr_critic_min must be <= lr_critic")
        
        # soft Q-function
        self.critic = QNetwork(args).to(self.device)
        self.critic_target = QNetwork(args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = opt.Adam(self.critic.parameters(), lr=args.lr_critic)
        
        # policy network (Gaussian only)
        print("\n---Gaussian policy is employed.---\n")
        self.actor = GaussianPolicy(args).to(self.device)
        self.actor_optimizer = opt.Adam(self.actor.parameters(), lr=args.lr_actor)
        
        # automatic_entropy_tuning
        if self.automatic_entropy_tuning is True:
            self.target_entropy = float(-args.action_dim)
            # self.target_entropy = log(args.action_dim)  # its value is 0, <float>    worse performance
            self.log_alpha = torch.ones(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = opt.Adam([self.log_alpha], lr=args.lr_alpha)
        self.lr_schedule_episodes = self._resolve_lr_schedule_episodes(args)
        self.lr_cycle_steps = self._resolve_lr_cycle_steps(self.lr_schedule_episodes)
        self.lr_warmup_steps = self._resolve_lr_warmup_steps(self.lr_cycle_steps)
        self.actor_lr_gamma = self._resolve_lr_gamma(
            max_lr=float(args.lr_actor),
            min_lr=self.actor_lr_min,
            total_episodes=self.lr_schedule_episodes,
            cycle_steps=self.lr_cycle_steps,
        )
        self.critic_lr_gamma = self._resolve_lr_gamma(
            max_lr=float(args.lr_critic),
            min_lr=self.critic_lr_min,
            total_episodes=self.lr_schedule_episodes,
            cycle_steps=self.lr_cycle_steps,
        )
        self.lr_schedule_step = 0
        self.critic_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=self.lr_cycle_steps,
            cycle_mult=1.0,
            max_lr=float(args.lr_critic),
            min_lr=self.critic_lr_min,
            warmup_steps=self.lr_warmup_steps,
            gamma=self.critic_lr_gamma,
        )
        self.actor_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=self.lr_cycle_steps,
            cycle_mult=1.0,
            max_lr=float(args.lr_actor),
            min_lr=self.actor_lr_min,
            warmup_steps=self.lr_warmup_steps,
            gamma=self.actor_lr_gamma,
        )
        # device info
        print('actor device: ', list(self.actor.parameters())[0].device)
        print('critic device: ', list(self.critic.parameters())[0].device)
        # print('alpha device: ', list(self.critic.parameters())[0].device)
        
        # create the directory to store the model (also needed when loading then saving)
        self.model_path = os.path.join(args.save_dir, args.scenario_name, "net_params")
        os.makedirs(self.model_path, exist_ok=True)
        self.checkpoint_prefix = self._normalize_ckpt_prefix(getattr(args, "checkpoint_prefix", ""))

        if args.load_or_not is True:
            # load model to evaluate
            load_path = os.path.join(args.load_dir, args.load_scenario_name, "net_params")
            loaded = False
            for prefix in self._candidate_ckpt_prefixes(args):
                load_a = os.path.join(load_path, self._ckpt_filename("actor", args.load_episode, prefix))
                load_c = os.path.join(load_path, self._ckpt_filename("critic", args.load_episode, prefix))
                if os.path.exists(load_a) and os.path.exists(load_c):
                    self.actor.load_state_dict(torch.load(load_a))
                    self.critic.load_state_dict(torch.load(load_c))
                    self.checkpoint_prefix = self._normalize_ckpt_prefix(prefix)
                    print('Agent successfully loaded actor_network: {}'.format(load_a))
                    print('Agent successfully loaded critic_network: {}'.format(load_c))
                    loaded = True
                    break
            if not loaded:
                print('----Failed to load----')
            if self.automatic_entropy_tuning is True:
                alpha_candidates = [self.checkpoint_prefix, ""]
                alpha_loaded = False
                for prefix in alpha_candidates:
                    load_alpha = os.path.join(load_path, self._ckpt_filename("alpha", args.load_episode, prefix))
                    if os.path.exists(load_alpha):
                        self.alpha_optimizer.load_state_dict(torch.load(load_alpha, map_location=self.device))
                        print('Agent successfully loaded alpha_network: {}'.format(load_alpha))
                        alpha_loaded = True
                        break
                if not alpha_loaded:
                    print('----alpha params not found, skip loading alpha----')

    @staticmethod
    def _normalize_ckpt_prefix(prefix):
        txt = str(prefix).strip()
        if not txt:
            return ""
        txt = txt.replace("-", "_")
        if re.fullmatch(r"\d+", txt):
            txt = f"iter{int(txt):03d}"
        m = re.search(r"iter[_-]?(\d+)", txt, flags=re.IGNORECASE)
        if m:
            txt = f"iter{int(m.group(1)):03d}"
        txt = re.sub(r"[^0-9A-Za-z_]", "", txt)
        if not txt:
            return ""
        if not txt.endswith("_"):
            txt += "_"
        return txt

    @staticmethod
    def _infer_iter_prefix_from_path(path):
        if not path:
            return ""
        for part in os.path.normpath(str(path)).split(os.sep):
            m = re.fullmatch(r"iter_(\d+)", part)
            if m:
                return f"iter{int(m.group(1)):03d}_"
        return ""

    @staticmethod
    def _resolve_lr_schedule_episodes(args):
        explicit = int(getattr(args, "lr_schedule_episodes", 0) or 0)
        if explicit > 0:
            return explicit
        chunk_episodes = int(getattr(args, "chunk_episodes", 0) or 0)
        if chunk_episodes > 0:
            return chunk_episodes
        total_episodes = int(getattr(args, "max_episodes", 0) or 0)
        if total_episodes > 0:
            return total_episodes
        return 1

    @staticmethod
    def _resolve_lr_cycle_steps(total_episodes):
        total_episodes = max(1, int(total_episodes))
        cycle_count = min(4, total_episodes)
        return max(1, math.ceil(total_episodes / cycle_count))

    @staticmethod
    def _resolve_lr_warmup_steps(cycle_steps):
        cycle_steps = max(1, int(cycle_steps))
        if cycle_steps <= 1:
            return 0
        return min(cycle_steps - 1, max(1, int(round(cycle_steps * 0.1))))

    @staticmethod
    def _resolve_lr_gamma(max_lr, min_lr, total_episodes, cycle_steps):
        if max_lr <= min_lr:
            return 1.0
        cycle_count = max(1, math.ceil(max(1, int(total_episodes)) / max(1, int(cycle_steps))))
        if cycle_count <= 1:
            return 1.0
        target_last_peak = min_lr + 0.1 * (max_lr - min_lr)
        return float((target_last_peak / max_lr) ** (1.0 / (cycle_count - 1)))

    def _candidate_ckpt_prefixes(self, args):
        prefixes = []

        def _add(v):
            p = self._normalize_ckpt_prefix(v)
            if p not in prefixes:
                prefixes.append(p)

        _add(getattr(args, "checkpoint_prefix", ""))
        _add(getattr(args, "eval_iter", ""))
        _add(self._infer_iter_prefix_from_path(getattr(args, "load_dir", "")))
        _add("")
        return prefixes

    def set_checkpoint_prefix(self, prefix):
        self.checkpoint_prefix = self._normalize_ckpt_prefix(prefix)

    def _ckpt_filename(self, key, episode, prefix=None):
        p = self.checkpoint_prefix if prefix is None else self._normalize_ckpt_prefix(prefix)
        if p:
            return f"{key}_{p}ep{int(episode)}.pkl"
        return f"{key}_params_ep{int(episode)}.pkl"

    def _best_filename(self, key, prefix=None):
        p = self.checkpoint_prefix if prefix is None else self._normalize_ckpt_prefix(prefix)
        if p:
            return f"{key}_{p}best.pkl"
        return f"{key}_best.pkl"

    def _best_meta_filename(self, prefix=None):
        p = self.checkpoint_prefix if prefix is None else self._normalize_ckpt_prefix(prefix)
        if p:
            return f"best_meta_{p.rstrip('_')}.json"
        return "best_meta.json"

    
    def _soft_update_target_network(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1-self.tau)*target_param.data+self.tau*param.data)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # Tensor(1,1)
        if evaluate is False:
            action, _, _ = self.actor.get_action(state)
        else:
            _, _, action = self.actor.get_action(state)
        return action.detach().cpu().numpy()[0]
    
    def learn(self, transition):
        state_batch = transition[0]
        action_batch = transition[1]
        reward_batch = transition[2]
        next_state_batch = transition[3]
        # mask_batch = transition[4]
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        # Update the Q-function parameters
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.get_action(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)-self.alpha*next_log_pi
            # next_q_value = reward_batch + (1-mask_batch)*self.gamma*min_qf_next_target
            next_q_value = reward_batch + self.gamma*min_qf_next_target
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss+qf2_loss
        
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()
        # soft update target Q-function parameters
        self._soft_update_target_network()
        
        # Update policy weights
        action, log_pi, _ = self.actor.get_action(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha*log_pi)-min_qf_pi).mean()
        # minimizing this: Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Adjust temperature
        if self.automatic_entropy_tuning is True:
            alpha_loss = -(self.log_alpha*(log_pi+self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs
        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def save_net(self, episode):
        torch.save(self.actor.state_dict(), os.path.join(self.model_path, self._ckpt_filename("actor", episode)))
        torch.save(self.critic.state_dict(), os.path.join(self.model_path, self._ckpt_filename("critic", episode)))
        if self.automatic_entropy_tuning is True:
            torch.save(
                self.alpha_optimizer.state_dict(),
                os.path.join(self.model_path, self._ckpt_filename("alpha", episode)),
            )
         
    def get_lrs(self):
        lrcr = self.critic_optimizer.param_groups[0]['lr']
        lrac = self.actor_optimizer.param_groups[0]['lr']
        lral = self.alpha_optimizer.param_groups[0]['lr'] if self.automatic_entropy_tuning else 0.0
        return lrcr, lrac, lral

    def step_lr_schedulers(self):
        self.lr_schedule_step += 1
        self.critic_scheduler.step(self.lr_schedule_step)
        self.actor_scheduler.step(self.lr_schedule_step)
        return self.get_lrs()

    def save_best(self, episode, reward):
        torch.save(self.actor.state_dict(), os.path.join(self.model_path, self._best_filename("actor")))
        torch.save(self.critic.state_dict(), os.path.join(self.model_path, self._best_filename("critic")))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(self.model_path, self._best_filename("actor_opt")))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(self.model_path, self._best_filename("critic_opt")))
        if self.automatic_entropy_tuning is True:
            torch.save(self.alpha_optimizer.state_dict(), os.path.join(self.model_path, self._best_filename("alpha")))
        meta = {'episode': int(episode), 'reward': float(reward)}
        with open(os.path.join(self.model_path, self._best_meta_filename()), 'w') as f:
            json.dump(meta, f, indent=2)
