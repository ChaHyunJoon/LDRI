import argparse

def get_args():
    parser = argparse.ArgumentParser("Soft Actor Critic Implementation")
    # Environment
    parser.add_argument("--max_episodes", type=int, default=400, help="number of episodes ")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="starting episode index for continued training")
    parser.add_argument("--episode_steps", type=int, default=None, help="number of time steps in a single episode")
    # Core training parameters
    parser.add_argument("--lr_critic", type=float, default=1e-3,
                        help="maximum learning rate of critic cosine annealing")
    parser.add_argument("--lr_critic_min", type=float, default=5e-4,
                        help="minimum learning rate of critic cosine annealing")
    parser.add_argument("--lr_actor", type=float, default=1e-3,
                        help="maximum learning rate of actor cosine annealing")
    parser.add_argument("--lr_actor_min", type=float, default=5e-5,
                        help="minimum learning rate of actor cosine annealing")
    parser.add_argument("--lr_alpha", type=float, default=5e-4, help="fixed learning rate of alpha")
    parser.add_argument(
        "--lr_schedule_episodes",
        type=int,
        default=0,
        help="cosine annealing cycle length in episodes; <=0 uses chunk_episodes if present, else max_episodes",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Temperature parameter α determines the relative importance of the entropy term against the reward')
    parser.add_argument('--auto_tune', type=bool, default=True)
    parser.add_argument("--tau", type=float, default=0.005, help="parameter for updating the target network")
    # memory buffer
    parser.add_argument("--buffer_size", type=int, default=int(1e6),
                        help="number of transitions can be stored in buffer")
    # 200ep: 3e4 for CLTCP_WVUSUB_WVUINTER,  1e4 for WVUCITY_HWFET
    parser.add_argument("--batch_size", type=int, default=256, help="number of episodes to optimize at the same time")
    # random seeds
    parser.add_argument("--random_seed", type=bool, default=True)
    # device
    parser.add_argument("--cuda", type=bool, default=True, help="True for GPU, False for CPU")
    # method
    parser.add_argument("--DRL", type=str, default='SAC', choices=['SAC'], help="training method")
    # for SAC
    parser.add_argument("--policy", type=str, default='Gaussian', choices=['Gaussian'], help="policy type")
    # for DP
    parser.add_argument("--soc_tol", type=float, default=0.0, help="terminal SOC tolerance")
    parser.add_argument("--w_terminal", type=float, default=1e6, help="terminal SOC penalty weight")
    parser.add_argument(
        "--dp_soc_chunk_size",
        type=int,
        default=128,
        help="SOC chunk size for 2D DP backward recursion; larger is faster but uses more RAM",
    )
    parser.add_argument(
        "--dp_prev_fc_state_count",
        type=int,
        default=0,
        help="number of previous-FC-power states for 2D DP; <=0 uses the full action-matched grid",
    )
    # about EMS, weight coefficient
    parser.add_argument("--w_soc", type=float, default=500, help="weight coefficient for SOC reward")
    parser.add_argument(
        "--eq_h2_batt_coef",
        type=float,
        default=0.0164,
        help="battery power to equivalent hydrogen coefficient: h2_batt = P_batt / 1000 * coef",
    )
    parser.add_argument("--soc0", type=float, default=0.6, help="initial value of SOC")
    parser.add_argument("--soc_target", type=float, default=0.6, help="terminal SOC target for DP")
    parser.add_argument("--MODE", type=str, default='CS', help="CS or CD, charge-sustain or charge-depletion")
    # save model under training     # Standard_ChinaCity  Standard_IM240
    parser.add_argument("--scenario_name", type=str, default="CLTC_P_plus_Standard_WVUINTER",
                        help="name of driving cycle data")
    # CLTC_P  Standard_ChinaCity, CLTCP_WVUSUB_WVUINTER, WVUCITY_HWFET, CLTCP_WVUINTER
    parser.add_argument("--save_dir", type=str, default="./test5", help="directory in which saves training data and model")
    parser.add_argument("--log_dir", type=str, default="./logs5", help="directory in which saves logs")
    parser.add_argument("--file_v", type=str, default='v1', help="version of driving cycle data file")
    # load learned model to train new model or evaluate
    parser.add_argument("--load_or_not", type=bool, default=True)
    parser.add_argument("--load_episode", type=int, default=397)
    parser.add_argument("--load_scenario_name", type=str, default="CLTCP_WVUINTER_w100_LR1e-03_v1")
    parser.add_argument("--load_dir", type=str, default="test2_SAC_CS_Gaussian")
    # evaluate
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--evaluate_episode", type=int, default=1)
    parser.add_argument("--eval_stochastic", type=bool, default=False,
                        help="if True, sample stochastic actions during evaluation")
    parser.add_argument(
        "--eval_iter",
        type=str,
        default="",
        help="optional LDRI iteration tag for evaluation output grouping (e.g., 1, 003, iter_003)",
    )
    parser.add_argument("--eva_dir", type=str, default="./eva")
    # checkpointing
    parser.add_argument("--save_best", type=bool, default=True, help="save best-performing episode checkpoint")
    # all above
    args = parser.parse_args()
    return args
    
