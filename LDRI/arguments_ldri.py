import argparse

from llm_presets import apply_agent_preset, get_agent_type_choices


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def get_ldri_args():
    parser = argparse.ArgumentParser("LDRI-style LLM reward refinement for FCEV")

    # Environment / task
    parser.add_argument("--scenario_name", type=str, default="Standard_WLTC_WVUINTER")
    parser.add_argument("--MODE", type=str, default="CS", help="CS or CD")
    parser.add_argument("--soc0", type=float, default=0.6)
    parser.add_argument("--soc_target", type=float, default=0.6)
    parser.add_argument("--w_soc", type=float, default=200.0)
    parser.add_argument("--w_h2", type=float, default=10.0)
    parser.add_argument("--w_fc", type=float, default=4.275e6)
    parser.add_argument("--w_batt", type=float, default=1.116e6)
    parser.add_argument("--eq_h2_batt_coef", type=float, default=0.0164)

    # DRL
    parser.add_argument("--DRL", type=str, default="SAC", choices=["SAC"])
    parser.add_argument("--policy", type=str, default="Gaussian", choices=["Gaussian"])
    parser.add_argument("--max_episodes", type=int, default=400)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--episode_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--auto_tune", type=str2bool, default=True)

    parser.add_argument("--lr_critic", type=float, default=7e-4)
    parser.add_argument("--lr_critic_min", type=float, default=2e-4)
    parser.add_argument("--lr_actor", type=float, default=5e-4)
    parser.add_argument("--lr_actor_min", type=float, default=5e-5)
    parser.add_argument("--lr_alpha", type=float, default=5e-4)
    parser.add_argument(
        "--lr_schedule_episodes",
        type=int,
        default=0,
        help="cosine annealing cycle length in episodes; <=0 uses chunk_episodes if present, else max_episodes",
    )

    # Device / seed
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=str2bool, default=True)

    # Loading prior checkpoint (optional)
    parser.add_argument("--load_or_not", type=str2bool, default=False)
    parser.add_argument("--load_episode", type=int, default=0)
    parser.add_argument("--load_scenario_name", type=str, default="")
    parser.add_argument("--load_dir", type=str, default="")

    # LDRI outer-loop
    parser.add_argument("--ldri_iterations", type=int, default=5)
    parser.add_argument(
        "--resume_run_root",
        type=str,
        default="",
        help="existing LDRI run root to resume; append iterations in-place",
    )
    parser.add_argument(
        "--resume_additional_iterations",
        type=int,
        default=0,
        help="extra iterations to run when resuming (if >0, overrides --ldri_iterations target)",
    )
    parser.add_argument(
        "--resume_use_saved_config",
        type=str2bool,
        default=True,
        help="when resuming, reuse run_config.json values to keep the original run conditions",
    )
    parser.add_argument(
        "--manual_pre_refine_source_iter",
        type=int,
        default=0,
        help=(
            "if >0, run LLM reward refinement for the target iteration using this "
            "existing iteration's train_log/episode_data before RL training"
        ),
    )
    parser.add_argument("--chunk_episodes", type=int, default=400)
    parser.add_argument("--feedback_process_episodes", type=int, default=10)
    parser.add_argument("--flush_buffer_on_patch", type=str2bool, default=True, 
                        help = 'when training with new reward, buffer is flushed')
    parser.add_argument(
        "--reset_agent_on_patch",
        type=str2bool,
        default=True,
        help="reset RL model parameters/optimizers after each successful reward patch",
    )
    parser.add_argument(
        "--reset_episode_counter_on_patch",
        type=str2bool,
        default=True,
        help="when resetting agent on patch, restart episode index from start_episode",
    )

    # Reward module and prompt template
    parser.add_argument("--reward_module", type=str, default="agentEMS_for_feedback")
    parser.add_argument("--reward_file", type=str, default="./agentEMS_for_feedback.py")
    parser.add_argument("--prompt_template", type=str, default="./reward_refinement_codex_prompt.md")

    # LLM client
    parser.add_argument(
        "--agent_type",
        type=str,
        default="gpt",
        choices=get_agent_type_choices(),
        help="high-level agent preset: gpt/openai, claude/anthropic, llama/huggingface",
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "huggingface", "openrouter"],
        help="optional provider override; defaults from --agent_type",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="optional model override; defaults from --agent_type",
    )
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument(
        "--llm_api_key_env",
        type=str,
        default=None,
        help="optional API-key env override; defaults from --agent_type",
    )
    parser.add_argument("--llm_temperature", type=float, default=0.2)
    parser.add_argument("--llm_max_tokens", type=int, default=3500)
    parser.add_argument("--llm_timeout_sec", type=int, default=240)
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--skip_patch", action="store_true")

    # Bootstrap (scratch reward generation before first training chunk)
    parser.add_argument(
        "--bootstrap_from_scratch",
        type=str2bool,
        default=True,
        help="generate an initial get_reward from environment/task description before LDRI iterations (required)",
    )
    parser.add_argument(
        "--bootstrap_only",
        type=str2bool,
        default=False,
        help="run only bootstrap reward generation/patch and exit before LDRI iterations",
    )
    parser.add_argument(
        "--bootstrap_prompt_template",
        type=str,
        default="./reward_generation_codex_prompt.md",
    )
    parser.add_argument("--bootstrap_retries", type=int, default=2)
    parser.add_argument(
        "--bootstrap_task_description",
        type=str,
        default="",
        help="optional extra task guidance text for scratch reward generation",
    )
    parser.add_argument("--bootstrap_max_attrs", type=int, default=80)
    parser.add_argument("--bootstrap_max_methods", type=int, default=40)

    # TPE-like gating
    parser.add_argument("--enable_tpe", type=str2bool, default=True)
    parser.add_argument(
        "--tpe_margin",
        type=float,
        default=0.05,
        help="primary preferred/dispreferred reward margin required to accept a patch",
    )
    parser.add_argument(
        "--parse_retries",
        type=int,
        default=8,
        help="extra retries reserved for parse failures before giving up",
    )
    parser.add_argument("--tpe_retries", type=int, default=10)
    parser.add_argument(
        "--tpe_soc_in_bounds_floor",
        type=float,
        default=100.0,
        help="minimum in_bounds_rate(%%) required when selecting both preferred and dispreferred episodes for TPE",
    )
    parser.add_argument(
        "--tpe_fp_pool_size",
        type=int,
        default=6,
        help="simple two-block TPE: feasible-episode pool size used to pick anchor/top/bottom groups",
    )
    parser.add_argument(
        "--tpe_fp_topk",
        type=int,
        default=2,
        help="simple two-block TPE: number of top feasible episodes used in the additional block",
    )
    parser.add_argument(
        "--tpe_fp_bottomk",
        type=int,
        default=2,
        help="simple two-block TPE: number of bottom feasible episodes used in the additional block",
    )
    parser.add_argument(
        "--tpe_fp_late_window",
        type=int,
        default=80,
        help="deprecated compatibility arg; ignored by the simple two-block TPE",
    )
    parser.add_argument(
        "--tpe_fp_recovery_margin",
        type=float,
        default=0.0,
        help="deprecated compatibility arg; ignored by the simple two-block TPE",
    )
    parser.add_argument(
        "--tpe_fp_guard_margin",
        type=float,
        default=0.0,
        help="simple two-block TPE: additional-block margin for top-pool vs bottom-pool mean reward",
    )
    parser.add_argument(
        "--tpe_fp_rank_corr_min",
        type=float,
        default=0.0,
        help="deprecated compatibility arg; ignored by the simple two-block TPE",
    )
    parser.add_argument(
        "--paper_battery_capacity_kwh",
        type=float,
        default=1.56,
        help="2024 paper-style equivalent-fuel battery capacity assumption in kWh",
    )
    parser.add_argument(
        "--paper_h2_lhv_kwh_per_kg",
        type=float,
        default=33.33,
        help="2024 paper-style equivalent-fuel hydrogen LHV assumption in kWh/kg",
    )
    parser.add_argument(
        "--paper_fc_efficiency",
        type=float,
        default=0.50,
        help="2024 paper-style equivalent-fuel fuel-cell efficiency assumption",
    )
    parser.add_argument(
        "--paper_electric_distance_km_per_kwh",
        type=float,
        default=5.0,
        help="2024 paper-style equivalent-fuel electric-distance correction assumption in km/kWh",
    )
    parser.add_argument(
        "--paper_dt_s",
        type=float,
        default=1.0,
        help="simulation step size used for 2024 paper-style equivalent-fuel replay in seconds",
    )
    parser.add_argument("--soc_low", type=float, default=0.4)
    parser.add_argument("--soc_high", type=float, default=0.8)

    # Output roots (separate from baseline)
    parser.add_argument("--ldri_root", type=str, default="./ldri_runs")
    parser.add_argument("--file_v", type=str, default="v_ldri")

    args = parser.parse_args()
    args = apply_agent_preset(args)
    return args
