import datetime
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as scio
import torch
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.memory import MemoryBuffer
from common.utils import summarize_fc_efficiency
from sac import SAC


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
    """FCEV Reference-style US MPG from H2 mass flow and battery power."""
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


@dataclass
class ChunkResult:
    iteration: int
    start_episode: int
    end_episode: int
    log_path: str
    episode_data_dir: str
    model_dir: str
    chunk_dir: str
    best_episode: Optional[int]
    best_reward: Optional[float]


class RunnerLDRI:
    """LDRI-style runner: train chunk episodes, then allow outer-loop reward refinement."""

    def __init__(self, args, env, run_root):
        self.args = args
        self.env = env
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.model_root = Path(self.args.save_dir).resolve()
        self.model_root.mkdir(parents=True, exist_ok=True)

        self.buffer = MemoryBuffer(args)
        self.SAC_agent = SAC(args)

        self.current_episode = int(getattr(args, "start_episode", 0))
        self.episode_step = int(args.episode_steps)
        self.DONE = {}

        self.seed = self._init_seed()
        self.best_reward = -float("inf")
        self.best_episode = None

    def _init_seed(self):
        if getattr(self.args, "random_seed", True):
            seed = int(np.random.randint(10000))
        else:
            seed = 93
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return seed

    def set_env(self, env):
        self.env = env

    def reset_buffer(self):
        self.buffer = MemoryBuffer(self.args)

    def reset_learning_state(self, reset_episode_counter: bool = True):
        """Reset agent parameters/optimizers and replay state for scratch retraining."""
        self.buffer = MemoryBuffer(self.args)
        self.DONE = {}
        self.best_reward = -float("inf")
        self.best_episode = None

        # Force fresh parameter initialization (do not reload checkpoints).
        old_load_or_not = bool(getattr(self.args, "load_or_not", False))
        old_evaluate = bool(getattr(self.args, "evaluate", False))
        try:
            self.args.load_or_not = False
            self.args.evaluate = False
            self.SAC_agent = SAC(self.args)
        finally:
            self.args.load_or_not = old_load_or_not
            self.args.evaluate = old_evaluate

        if reset_episode_counter:
            self.current_episode = int(getattr(self.args, "start_episode", 0))

    @staticmethod
    def _episode_info_template():
        return {
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
            "soc_in_bounds": [],
        }

    @staticmethod
    def _append_info(episode_info, info):
        for key in episode_info.keys():
            episode_info[key].append(info.get(key, np.nan))

    def _open_chunk_log(self, chunk_dir: Path):
        chunk_dir.mkdir(parents=True, exist_ok=True)
        log_path = chunk_dir / "train_log.log"
        return log_path, log_path.open("w", encoding="utf-8")

    # model을 저장할 directory 생성
    def _configure_model_dir(self, iteration: int) -> Path:
        iter_save_dir = self.model_root / f"iter_{int(iteration):03d}"
        model_dir = iter_save_dir / self.args.scenario_name / "net_params"
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_prefix = f"iter{int(iteration):03d}"

        # Keep args and agents aligned so checkpoints are written per LDRI iteration.
        self.args.save_dir = str(iter_save_dir)
        self.args.checkpoint_prefix = ckpt_prefix
        self.SAC_agent.model_path = str(model_dir)
        self.SAC_agent.set_checkpoint_prefix(ckpt_prefix)
        return model_dir

    @staticmethod
    def _log(log_file, msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    @staticmethod
    def _fmt_hms(seconds):
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _finalize_episode_metrics(self, episode_info, info, episode_reward):
        travel_m = float(info["travel"])
        travel = travel_m / 1000.0
        h2 = float(np.sum(episode_info["h2_fcs"]))
        h2_100 = h2 / travel * 100.0 if travel > 1e-9 else np.nan
        h2_batt = float(np.nansum(np.asarray(episode_info.get("h2_batt", []), dtype=np.float64)))
        h2_batt_100 = h2_batt / travel * 100.0 if travel > 1e-9 else np.nan
        h2_equal = float(np.nansum(np.asarray(episode_info.get("h2_equal", []), dtype=np.float64)))
        h2_equal_100 = h2_equal / travel * 100.0 if travel > 1e-9 else np.nan
        mpg, fuel_eq_total_gal = compute_fcev_reference_mpg(
            travel_m=travel_m,
            h2_fcs_gps=episode_info["h2_fcs"],
            p_batt_kw=episode_info["P_batt_req"],
            dt_s=float(getattr(self.env.agent.FCHEV, "simtime", 1.0)),
        )

        objective_cost = float(np.sum(episode_info["objective_cost"]))
        objective_100 = objective_cost / travel * 100.0 if travel > 1e-9 else np.nan

        soc_arr = np.asarray(episode_info["SOC"], dtype=np.float64)
        soc_min = float(np.min(soc_arr))
        soc_max = float(np.max(soc_arr))
        soc_mean = float(np.mean(soc_arr))

        soc_in_bounds = np.asarray(episode_info["soc_in_bounds"], dtype=np.float64)
        soc_in_bounds_rate = float(np.mean(soc_in_bounds)) * 100.0

        mean_soc_cost = float(np.mean(episode_info["soc_cost"]))
        mean_h2_cost = float(np.mean(episode_info["h2_cost"]))
        mean_fcs_cost = float(np.mean(episode_info["fcs_soh_cost"]))
        mean_batt_cost = float(np.mean(episode_info["batt_soh_cost"]))
        mean_objective = float(np.mean(episode_info["objective_cost"]))

        p_fc = np.asarray(episode_info["P_fc"], dtype=np.float64)
        p_batt = np.asarray(episode_info["P_batt_req"], dtype=np.float64)
        min_p_fc = float(np.min(p_fc))
        mean_p_fc = float(np.mean(p_fc))
        max_p_fc = float(np.max(p_fc))
        min_p_batt = float(np.min(p_batt))
        mean_p_batt = float(np.mean(p_batt))
        max_p_batt = float(np.max(p_batt))

        mean_fce_eff, pct_fc_on = summarize_fc_efficiency(
            p_fc,
            episode_info["fce_eff"],
            on_threshold=self.env.agent.FCHEV.P_FC_off,
        )

        ep_len = len(episode_reward)
        ep_r = float(np.sum(episode_reward)) if episode_reward else 0.0

        return {
            "travel": travel,
            "h2_100": h2_100,
            "h2_batt_100": h2_batt_100,
            "h2_equal_100": h2_equal_100,
            "mpg": mpg,
            "fuel_eq_total_gal": fuel_eq_total_gal,
            "objective_100": objective_100,
            "soc_min": soc_min,
            "soc_mean": soc_mean,
            "soc_max": soc_max,
            "soc_in_bounds_rate": soc_in_bounds_rate,
            "mean_soc_cost": mean_soc_cost,
            "mean_h2_cost": mean_h2_cost,
            "mean_fcs_cost": mean_fcs_cost,
            "mean_batt_cost": mean_batt_cost,
            "mean_objective": mean_objective,
            "min_p_fc": min_p_fc,
            "mean_p_fc": mean_p_fc,
            "max_p_fc": max_p_fc,
            "min_p_batt": min_p_batt,
            "mean_p_batt": mean_p_batt,
            "max_p_batt": max_p_batt,
            "mean_fce_eff": mean_fce_eff,
            "pct_fc_on": pct_fc_on,
            "ep_len": ep_len,
            "ep_r": ep_r,
            "soc": float(info["SOC"]),
            "fcs_soh": float(info["FCS_SOH"]),
            "bat_soh": float(info["SOH"]),
        }

    def _save_episode_data(self, episode_idx, episode_info, episode_data_dir: Path):
        if "alpha" not in episode_info:
            episode_info["alpha"] = []
        datadir = episode_data_dir / f"data_ep{episode_idx}.mat"
        scio.savemat(str(datadir), mdict=episode_info)

    def _run_one_episode_sac(self, episode_idx, episode_data_dir):
        state = self.env.reset()
        episode_reward = []
        c_loss_1_one_ep = []
        c_loss_2_one_ep = []
        a_loss_one_ep = []
        en_loss_one_ep = []
        alpha_value_ep = []
        episode_info = self._episode_info_template()

        for episode_step in range(self.episode_step):
            with torch.no_grad():
                action = self.SAC_agent.select_action(state, evaluate=False)
            state_next, step_reward, done, info = self.env.step(action, episode_step)
            self.buffer.store(state, action, step_reward, state_next, done)
            state = state_next

            self._append_info(episode_info, info)
            episode_reward.append(step_reward)

            if self.buffer.currentSize >= 10 * self.args.batch_size:
                transition = self.buffer.random_sample()
                c1, c2, a, en, alpha = self.SAC_agent.learn(transition)
                c_loss_1_one_ep.append(c1)
                c_loss_2_one_ep.append(c2)
                a_loss_one_ep.append(a)
                en_loss_one_ep.append(en)
                alpha_value_ep.append(alpha)

            if done and episode_idx not in self.DONE:
                self.DONE[episode_idx] = episode_step

        episode_info["alpha"] = alpha_value_ep
        self.SAC_agent.save_net(episode_idx)
        self._save_episode_data(episode_idx, episode_info, episode_data_dir)

        lrcr, lrac, lral = self.SAC_agent.get_lrs()
        metrics = self._finalize_episode_metrics(episode_info, info, episode_reward)

        ep_c1 = float(np.mean(c_loss_1_one_ep)) if c_loss_1_one_ep else 0.0
        ep_c2 = float(np.mean(c_loss_2_one_ep)) if c_loss_2_one_ep else 0.0
        ep_a = float(np.mean(a_loss_one_ep)) if a_loss_one_ep else 0.0
        ep_en = float(np.mean(en_loss_one_ep)) if en_loss_one_ep else 0.0
        ep_alpha = float(np.mean(alpha_value_ep)) if alpha_value_ep else float(self.SAC_agent.alpha)

        metrics.update(
            {
                "ep_c1": ep_c1,
                "ep_c2": ep_c2,
                "ep_a": ep_a,
                "ep_en": ep_en,
                "ep_alpha": ep_alpha,
                "lrcr": lrcr,
                "lrac": lrac,
                "lral": lral,
            }
        )
        if self.SAC_agent is not None:
            self.SAC_agent.step_lr_schedulers()
        return metrics

    def _maybe_update_best(self, episode_idx, ep_reward):
        if ep_reward > self.best_reward:
            self.best_reward = float(ep_reward)
            self.best_episode = int(episode_idx)
            if self.SAC_agent is not None:
                self.SAC_agent.save_best(episode_idx, ep_reward)
            return True
        return False

    def train_chunk(self, iteration, chunk_episodes):
        chunk_episodes = int(chunk_episodes)
        chunk_dir = self.run_root / f"iter_{int(iteration):03d}"
        episode_data_dir = chunk_dir / "episode_data"
        episode_data_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self._configure_model_dir(iteration)
        log_path, log_file = self._open_chunk_log(chunk_dir)

        self._log(log_file, f"LDRI chunk iteration: {iteration}")
        self._log(log_file, f"time: {datetime.datetime.now().isoformat()}")
        self._log(log_file, f"DRL: {self.args.DRL}, seed: {self.seed}")
        self._log(log_file, f"model_dir: {model_dir}")
        self._log(log_file, f"chunk_episodes: {chunk_episodes}, episode_steps: {self.episode_step}")

        start_episode = self.current_episode
        chunk_start_time = time.time()
        with tqdm(total=chunk_episodes, desc=f"LDRI iter {int(iteration):03d}", unit="ep") as pbar:
            for ep_i in range(chunk_episodes):
                episode_idx = self.current_episode

                m = self._run_one_episode_sac(episode_idx, episode_data_dir)
                self._log(
                    log_file,
                    "\nepi %d: travel %.3fkm, SOC %.4f, FCS-SOH %.6f, Bat-SOH %.6f"
                    % (episode_idx, m["travel"], m["soc"], m["fcs_soh"], m["bat_soh"]),
                )
                self._log(
                    log_file,
                    "epi %d: H2_100km %.1fg, H2eq_100km %.1fg (batt %.1fg), MPG %.2f, objective_100km %.2f"
                    % (episode_idx, m["h2_100"], m["h2_equal_100"], m["h2_batt_100"], m["mpg"], m["objective_100"]),
                )
                self._log(
                    log_file,
                    "epi %d: SOC[min/mean/max]=%.3f/%.3f/%.3f, in_bounds_rate=%.1f%%, len=%d"
                    % (
                        episode_idx,
                        m["soc_min"],
                        m["soc_mean"],
                        m["soc_max"],
                        m["soc_in_bounds_rate"],
                        m["ep_len"],
                    ),
                )
                self._log(
                    log_file,
                    "epi %d: mean_cost soc %.3f, h2 %.3f, fcs %.3f, batt %.3f, obj %.3f"
                    % (
                        episode_idx,
                        m["mean_soc_cost"],
                        m["mean_h2_cost"],
                        m["mean_fcs_cost"],
                        m["mean_batt_cost"],
                        m["mean_objective"],
                    ),
                )
                self._log(
                    log_file,
                    "epi %d: P_fc min/mean/max %.2f/%.2f/%.2f kW, P_batt_req min/mean/max %.2f/%.2f/%.2f kW, fce_eff mean %.3f, fc_on %.1f%%"
                    % (
                        episode_idx,
                        m["min_p_fc"],
                        m["mean_p_fc"],
                        m["max_p_fc"],
                        m["min_p_batt"],
                        m["mean_p_batt"],
                        m["max_p_batt"],
                        m["mean_fce_eff"],
                        m["pct_fc_on"],
                    ),
                )
                self._log(
                    log_file,
                    "epi %d: ep_cumulative_r %.3f, c-loss1 %.4f, c-loss2 %.4f, a-loss %.4f, en-loss %.4f, alpha %.6f"
                    % (
                        episode_idx,
                        m["ep_r"],
                        m["ep_c1"],
                        m["ep_c2"],
                        m["ep_a"],
                        m["ep_en"],
                        m["ep_alpha"],
                    ),
                )
                self._log(
                    log_file,
                    "epi %d: lr_critic %.6f, lr_actor %.6f, lr_alpha %.6f"
                    % (episode_idx, m["lrcr"], m["lrac"], m["lral"]),
                )

                if self._maybe_update_best(episode_idx, m["ep_r"]):
                    self._log(log_file, f"best checkpoint updated: ep {episode_idx}, reward {m['ep_r']:.3f}")

                self.current_episode += 1
                pbar.update(1)
                pbar.set_postfix(ep=episode_idx, r=f"{m['ep_r']:.1f}", obj100=f"{m['objective_100']:.1f}", mpg=f"{m['mpg']:.1f}")

                elapsed = time.time() - chunk_start_time
                done_eps = ep_i + 1
                avg_sec = elapsed / done_eps if done_eps > 0 else 0.0
                remaining_eps = chunk_episodes - done_eps
                eta_sec = avg_sec * remaining_eps
                self._log(
                    log_file,
                    "chunk_progress: %d/%d, elapsed=%s, eta=%s, avg_sec_per_ep=%.1f"
                    % (
                        done_eps,
                        chunk_episodes,
                        self._fmt_hms(elapsed),
                        self._fmt_hms(eta_sec),
                        avg_sec,
                    ),
                )

        end_episode = self.current_episode - 1
        self._log(log_file, f"chunk finished: episodes {start_episode}..{end_episode}")
        self._log(log_file, f"buffer current size: {self.buffer.currentSize}, counter: {self.buffer.counter}")
        log_file.close()

        return ChunkResult(
            iteration=int(iteration),
            start_episode=int(start_episode),
            end_episode=int(end_episode),
            log_path=str(log_path),
            episode_data_dir=str(episode_data_dir),
            model_dir=str(model_dir),
            chunk_dir=str(chunk_dir),
            best_episode=self.best_episode,
            best_reward=(None if self.best_reward == -float("inf") else self.best_reward),
        )
