import importlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import get_driving_cycle, get_acc_limit


class EnvLDRI:
    """LDRI workflow environment that can reload reward module between iterations."""

    def __init__(self, args, reward_module="agentEMS_for_feedback", force_reload_agent=False):
        self.args = args
        self.speed_list = get_driving_cycle(cycle_name=args.scenario_name)
        self.acc_list = get_acc_limit(self.speed_list, output_max_min=False)
        self.abs_spd_MAX = max(abs(self.speed_list))
        self.abs_acc_MAX = max(abs(max(self.acc_list)), abs(min(self.acc_list)))

        EMS = self._load_ems_class(reward_module=reward_module, force_reload=force_reload_agent)
        self.agent = EMS(
            args.w_soc,
            args.soc0,
            args.MODE,
            self.abs_spd_MAX,
            self.abs_acc_MAX,
            w_h2=args.w_h2,
            w_fc=args.w_fc,
            w_batt=args.w_batt,
            eq_h2_batt_coef=args.eq_h2_batt_coef,
            soc_target=args.soc_target,
            soc_bounds=(args.soc_low, args.soc_high),
        )
        self.obs_num = self.agent.obs_num
        self.action_num = self.agent.action_num

    @staticmethod
    def _load_ems_class(reward_module, force_reload=False):
        module = importlib.import_module(reward_module)
        if force_reload:
            module = importlib.reload(module)
        if not hasattr(module, "EMS"):
            raise AttributeError(f"module '{reward_module}' does not define EMS class")
        return module.EMS

    def reset(self):
        return self.agent.reset_obs()

    def step(self, action, episode_step):
        car_spd = self.speed_list[episode_step]
        car_acc = self.acc_list[episode_step]

        obs = self.agent.execute(action, car_spd, car_acc)
        reward = self.agent.get_reward()
        done = self.agent.get_done()
        info = self.agent.get_info()

        return obs, reward, done, info


def make_env_ldri(args, force_reload_agent=True):
    env = EnvLDRI(
        args,
        reward_module=getattr(args, "reward_module", "agentEMS_for_feedback"),
        force_reload_agent=force_reload_agent,
    )
    args.obs_dim = env.agent.obs_num
    args.action_dim = env.agent.action_num
    args.episode_steps = len(env.speed_list)
    args.abs_spd_MAX = env.abs_spd_MAX
    args.abs_acc_MAX = env.abs_acc_MAX
    return env, args
