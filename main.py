import sys
import warnings
import json
import re
from pathlib import Path
from common.runner import Runner
from common.arguments import get_args
from common.env import make_env
from common.evaluate import Evaluator
from common.utils import Logger

warnings.filterwarnings("ignore")


def _format_iter_tag(value):
    txt = str(value).strip()
    if not txt:
        return None
    m = re.search(r"(\d+)", txt)
    if not m:
        return None
    return f"iter_{int(m.group(1)):03d}"


def _infer_iter_from_load_dir(load_dir):
    if not load_dir:
        return None
    for part in Path(load_dir).parts:
        m = re.fullmatch(r"iter_(\d+)", part)
        if m:
            return f"iter_{int(m.group(1)):03d}"
    return None


def _infer_iter_from_run_summary(load_dir, load_episode):
    if not load_dir:
        return None
    p = Path(load_dir).resolve()
    candidates = [p] + list(p.parents[:5])
    for c in candidates:
        summary_path = c / "run_summary.json"
        if not summary_path.exists():
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue
        matches = []
        for rec in summary.get("iterations", []):
            chunk = rec.get("chunk", {})
            start_ep = chunk.get("start_episode")
            end_ep = chunk.get("end_episode")
            if start_ep is None or end_ep is None:
                continue
            if int(start_ep) <= int(load_episode) <= int(end_ep):
                iter_idx = rec.get("iteration")
                if iter_idx is not None:
                    matches.append(int(iter_idx))
        if len(matches) == 1:
            return f"iter_{matches[0]:03d}"
    return None


def _resolve_eval_iter_tag(args):
    explicit = _format_iter_tag(getattr(args, "eval_iter", ""))
    if explicit is not None:
        return explicit
    from_dir = _infer_iter_from_load_dir(getattr(args, "load_dir", ""))
    if from_dir is not None:
        return from_dir
    return _infer_iter_from_run_summary(
        getattr(args, "load_dir", ""),
        int(getattr(args, "load_episode", 0)),
    )


if __name__ == '__main__':
    args = get_args()
    eval_cycle_name = args.scenario_name
    args.save_dir = args.save_dir + "_" + args.DRL + "_" + args.MODE + "_" + args.policy
    args.log_dir = args.log_dir + "_" + args.DRL + "_" + args.MODE + "_" + args.policy
    args.eva_dir = args.eva_dir + "_" + args.DRL + "_" + args.MODE + "_" + args.policy
    env, args = make_env(args)
    if args.evaluate:
        args.eval_cycle = eval_cycle_name
        args.eval_agent = args.load_scenario_name + '_%d' % args.load_episode
        args.eval_iter_tag = _resolve_eval_iter_tag(args)
        if args.eval_iter_tag:
            args.eval_name = args.eval_cycle + '/' + args.eval_iter_tag + '/' + args.eval_agent
        else:
            args.eval_name = args.eval_cycle + '/' + args.eval_agent
        sys.stdout = Logger(filepath=args.eva_dir+"/"+args.eval_name+"/", filename='evaluate_log.log')
        print('max_episodes: ', args.evaluate_episode)
    else:
        args.scenario_name = args.scenario_name + "_w%d"%args.w_soc + "_LR%.0e"%args.lr_critic + '_' + args.file_v
        sys.stdout = Logger(filepath=args.save_dir+"/"+args.scenario_name+"/", filename='train_log.log')
        print('\nweight coefficient: w_soc = %.1f' % args.w_soc)
        print('max_episodes: ', args.max_episodes)
    if args.evaluate:
        print('cycle name: ', args.eval_cycle)
        print('agent name: ', args.eval_agent)
        print('iteration tag: ', args.eval_iter_tag if args.eval_iter_tag else 'N/A')
        print('eval stochastic: ', getattr(args, "eval_stochastic", False))
    else:
        print('cycle name: ', args.scenario_name)
    print('episode_steps: ', args.episode_steps)
    print('abs_spd_MAX: %.3f m/s' % args.abs_spd_MAX)
    print('abs_acc_MAX: %.3f m/s2' % args.abs_acc_MAX)
    print("DRL method: ", args.DRL)
    print('obs_dim: ', args.obs_dim)
    print('action_dim: ', args.action_dim)
    print('critic warmup-restart cosine learning rate: %.0e -> %.0e' % (args.lr_critic, args.lr_critic_min))
    print('actor warmup-restart cosine learning rate: %.0e -> %.0e'% (args.lr_actor, args.lr_actor_min))
    if args.DRL == 'SAC':
        print('alpha fixed learning rate: %.0e'%args.lr_alpha)
    print('initial SOC: ', args.soc0)
    print('SOC-MODE: ', args.MODE)
    print('eq_h2_batt_coef: ', args.eq_h2_batt_coef)
    
    if args.evaluate:
        print("\n-----Start evaluating!-----")
        evaluator = Evaluator(args, env)
        evaluator.evaluate()
        print("-----Evaluating is finished!-----")
        print('-----Data saved in: <%s>-----'%(args.eva_dir+"/"+args.eval_name))
    else:
        print("\n-----Start training-----")
        runner = Runner(args, env)
        runner.set_seed()
        runner.run_SAC()
        runner.memory_info()
        print("-----Training is finished!-----")
        print('-----Data saved in: <%s>-----'%(args.save_dir+"/"+args.scenario_name))
