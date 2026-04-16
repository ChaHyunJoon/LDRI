# running environment of DP
import numpy as np
from common.utils import get_driving_cycle, get_acc_limit


class DP_Env:
    def __init__(self, scenario_name):
        # speed list
        self.speed_list = get_driving_cycle(cycle_name=scenario_name)
        self.acc_list = get_acc_limit(self.speed_list, output_max_min=False)
        self.time_steps = len(self.speed_list)
        # state space
        self.state_increment = 0.0001        # 1e-4, Battery soc를 4000개 구간으로 discretize
        self.state_init = 0.6
        self.state_max = 0.8
        self.state_min = 0.4
        self.states = np.arange(self.state_min, self.state_max, self.state_increment)
        # action space
        self.action_number = 95
        self.actions = np.linspace(-1.0, 1.0, self.action_number + 1, dtype=np.float32)
    
    
