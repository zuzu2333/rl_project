import os
import time
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from distutils.util import strtobool
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class EnsureActionDim1(gym.ActionWrapper):
    """确保连续动作环境收到的 action 形状为 (1,), 避免被 squeeze 成标量"""
    def action(self, action):
        a = np.asarray(action, dtype=np.float32)
        if a.shape == ():                # 标量 -> (1,)
            a = np.array([a], dtype=np.float32)
        elif a.shape == (1,) or a.shape == ():
            pass
        elif a.ndim > 1:
            a = a.reshape(-1)[:1]        # 兜底裁成(1,)
        return a

import re

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str) initial learning rate.
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining):
        return progress_remaining * initial_value

    return func

def make_lr(v):
    if isinstance(v, (float, int)):
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(r"linear_schedule\(([\deE\+\-\.]+)\)", v.replace(" ", ""))
        if m:
            return linear_schedule(float(m.group(1)))
    raise ValueError(f"Unsupported learning_rate format: {v}")

import time

class StateTrajectoryCallback(BaseCallback):
    """
    在若干里程碑步数触发一次评估，记录 MountainCarContinuous 轨迹并画图。
    - milestones: [1000, 5000, 20000] 这样的步数阈值
    - save_dir: 本地保存 png/csv 的目录
    - log_wandb: 若为 True 且 wandb.run 不为空，则把图与原始数组同步到 W&B
    依赖：eval_env 与训练用 VecNormalize 共享统计 (eval_env.training=False, eval_env.obs_rms=env.obs_rms)
    """
    def __init__(self, eval_env, milestones=(1000, 5000, 20000), save_dir="./artifacts/traj", log_wandb=True):
        super().__init__()
        self.eval_env = eval_env
        self.milestones = sorted(list(milestones))
        self.next_idx = 0
        self.save_dir = save_dir
        self.log_wandb = log_wandb
        os.makedirs(self.save_dir, exist_ok=True)

    def _run_one_episode(self):
        """deterministic rollout，返回 dict: t, pos, vel, act"""
        obs = self.eval_env.reset()
        t, pos, vel, act = [], [], [], []
        step = 0
        done = False  # 或 np.zeros(self.eval_env.num_envs, dtype=bool)
        while not done:        # 单环境 VecEnv
            # obs shape: (1, 2) for MountainCarContinuous -> [position, velocity]
            a, _ = self.model.predict(obs, deterministic=True)
            obs, r, dones, infos = self.eval_env.step(a)
            o = obs[0]  # (2,)
            t.append(step)
            pos.append(float(o[0]))
            vel.append(float(o[1]))
            act.append(float(a[0][0]) if np.ndim(a) == 2 else float(a[0]))
            step += 1
            done = bool(dones[0])
        return {"t": np.array(t), "position": np.array(pos),
                "velocity": np.array(vel), "action": np.array(act)}

    def _plot_and_save(self, data, tag):
        """保存 png 与 csv；必要时同步到 W&B"""
        png_path = os.path.join(self.save_dir, f"{tag}.png")
        csv_path = os.path.join(self.save_dir, f"{tag}.csv")

        # 保存 CSV（便于之后复现画图）
        arr = np.stack([data["t"], data["position"], data["velocity"], data["action"]], axis=1)
        np.savetxt(csv_path, arr, delimiter=",", header="t,position,velocity,action", comments="")

        # 画图（time vs pos/vel；action 作为副图可选）
        plt.figure(figsize=(8, 4))
        plt.plot(data["t"], data["position"], label="position")
        plt.plot(data["t"], data["velocity"], label="velocity")
        plt.xlabel("time step")
        plt.ylabel("state")
        plt.title(tag)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        # 同步到 W&B（若已初始化）
        if self.log_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(data=list(arr), columns=["t", "position", "velocity", "action"])
                    wandb.log({
                        "trajectory/image": wandb.Image(png_path),
                        "trajectory/table": table,
                        "trajectory/tag": tag
                    })
            except Exception:
                pass

        return png_path, csv_path

    def _on_step(self) -> bool:
        if self.next_idx < len(self.milestones) and self.num_timesteps >= self.milestones[self.next_idx]:
            tag = f"traj_step_{self.milestones[self.next_idx]}"
            data = self._run_one_episode()
            self._plot_and_save(data, tag)
            self.next_idx += 1
        return True

class EvalWithSuccessCallback(BaseCallback):
    """
    每 eval_freq 步，在 eval_env 上评估：
      - eval/mean_reward, eval/mean_ep_length
      - eval/success_rate  (MountainCarContinuous: 到达旗子为成功)
    """
    def __init__(self, eval_env, milestones=(1000, 5000, 20000), n_eval_episodes: int = 5):
        super().__init__()
        self.eval_env = eval_env
        self.milestones = sorted(list(milestones))
        self.next_milestone = 0
        self.n_eval_episodes = n_eval_episodes
        # 兼容 MountainCarContinuous：优先读 env.unwrapped.goal_position
        self._goal_pos = None

    def _on_training_start(self) -> None:
        try:
            self._goal_pos = getattr(self.eval_env.envs[0].unwrapped, "goal_position", 0.45)
        except Exception:
            self._goal_pos = 0.45

    def _do_eval(self):
        ep_rewards, ep_lengths, successes = [], [], []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()              # ← 只有 obs
            total_r, length = 0.0, 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)   # ← 4 元组
                total_r += float(rewards[0])
                length += 1
                done = bool(dones[0])

                if done:
                    info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                    term_obs = (
                        info0.get("terminal_observation", None)
                        if isinstance(info0, dict) else None
                    )
                    if term_obs is None and isinstance(info0, dict):
                        term_obs = info0.get("final_observation", None)

                    if term_obs is not None:
                        term = np.asarray(term_obs)

                        # 反归一化回原始状态（VecNormalize 自带）
                        if hasattr(self.eval_env, "unnormalize_obs"):
                            term = self.eval_env.unnormalize_obs(term)

                        # term 形状可能是 (1, 2) 或 (2,)
                        pos = float(term[0] if term.ndim == 1 else term[0, 0])

                        goal = getattr(self.eval_env.envs[0].unwrapped, "goal_position", 0.45)
                        success = 1.0 if pos >= goal else 0.0
                    else:
                        success = 1.0 if total_r >= 90.0 else 0.0
                        
            ep_rewards.append(total_r)
            ep_lengths.append(length)
            successes.append(success)

        self.logger.record("eval/mean_reward", float(np.mean(ep_rewards)))
        self.logger.record("eval/mean_ep_length", float(np.mean(ep_lengths)))
        self.logger.record("eval/success_rate", float(np.mean(successes)))

    def _on_step(self) -> bool:
        # 里程碑强制评估（确保 1k/5k/20k 正好有点，不用等到 rollout 结束）
        while self.next_milestone < len(self.milestones) and \
              self.num_timesteps >= self.milestones[self.next_milestone]:
            self._do_eval()
            self.next_milestone += 1
        return True

    def _on_rollout_end(self) -> None:
        # 每个 rollout 结束再评一次，和参数更新对齐
        self._do_eval()

class CustomMetricsCallback(BaseCallback):
    def __init__(self, noise_level=None, variant=None, hist_window_episodes=100):
        super().__init__()
        self.noise_level = noise_level
        self.variant = variant
        self._rewards = deque(maxlen=hist_window_episodes)
        self._lengths = deque(maxlen=hist_window_episodes)
        self._last_wall_time = None
        self._last_steps_for_wall = 0
        self._last_sps_time = None
        self._last_sps_steps = 0
        self._last_step_time = None
        self._ema_infer = None

    def _on_training_start(self) -> None:
        if self.noise_level is not None:
            self.logger.record("noise/level", float(self.noise_level))
        if self.variant is not None:
            self.logger.record("noise/variant", str(self.variant))
        now = time.time()
        self._last_wall_time = now
        self._last_steps_for_wall = self.num_timesteps
        self._last_sps_time = now
        self._last_sps_steps = self.num_timesteps
        self._last_step_time = now

    def _log_hist(self, name, arr):
        self.logger.record(name, arr)
        q10, q50, q90 = np.percentile(arr, [10, 50, 90])
        self.logger.record(f"{name}_p10", float(q10))
        self.logger.record(f"{name}_p50", float(q50))
        self.logger.record(f"{name}_p90", float(q90))
        self.logger.record(f"{name}_std", float(arr.std(ddof=0)))

    def _on_step(self) -> bool:
        now = time.time()

        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                ep = info.get("episode")
                if ep is not None:
                    self._rewards.append(float(ep["r"]))
                    self._lengths.append(float(ep["l"]))
                    if len(self._rewards) == self._rewards.maxlen:
                        self._log_hist("charts/episodic_return_raw", np.asarray(self._rewards, np.float32))
                        self._log_hist("charts/episodic_length_raw", np.asarray(self._lengths, np.float32))

        # SPS
        sps_dt = now - self._last_sps_time
        sps_dsteps = self.num_timesteps - self._last_sps_steps
        if sps_dt > 0:
            self.logger.record("charts/SPS", float(sps_dsteps / sps_dt))
            self._last_sps_time = now
            self._last_sps_steps = self.num_timesteps

        # wallclock per 1k steps
        dsteps = self.num_timesteps - self._last_steps_for_wall
        if dsteps >= 1000:
            dt = now - self._last_wall_time
            if dt > 0:
                self.logger.record("time/wallclock_per_1k_steps", float(dt * (1000.0 / dsteps)))
            self._last_wall_time = now
            self._last_steps_for_wall = self.num_timesteps

        # inference time (approx)
        dt = now - self._last_step_time
        steps = max(1, self.locals.get("n_steps", 1))
        est = dt / steps
        alpha = 0.1
        self._ema_infer = est if self._ema_infer is None else (1 - alpha) * self._ema_infer + alpha * est
        self.logger.record("perf/inference_time_per_step", float(self._ema_infer))
        self._last_step_time = now

        return True



@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    print("Final config:\n", OmegaConf.to_yaml(cfg))

    run_name = f"{cfg.env.env_id}_ppo_s{cfg.seed}_{int(time.time())}"
    os.makedirs(cfg.save_path, exist_ok=True)

    if cfg.wandb.track:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            sync_tensorboard=True,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
        )

    vec_cls = SubprocVecEnv if cfg.vec_backend == "subproc" else DummyVecEnv
    
    # --- train env
    env = make_vec_env(cfg.env.env_id, n_envs=cfg.n_envs, seed=cfg.seed,wrapper_class=EnsureActionDim1, 
                    vec_env_cls=SubprocVecEnv if cfg.vec_backend=="subproc" else DummyVecEnv)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- eval env
    eval_env = make_vec_env(cfg.env.env_id, n_envs=1, seed=cfg.seed+10, wrapper_class=EnsureActionDim1, vec_env_cls=DummyVecEnv)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.obs_rms = env.obs_rms  # 共享 obs 统计
    eval_env.training = False

    model = PPO(
    policy=cfg.ppo.policy,
    env=env,
    learning_rate=make_lr(cfg.ppo.learning_rate),           # 可做线性退火: schedule
    gamma=cfg.ppo.gamma,
    gae_lambda=cfg.ppo.gae_lambda,
    clip_range=cfg.ppo.clip_range,
    clip_range_vf=0.2,
    ent_coef=cfg.ppo.ent_coef,                     # 可设 0.005~0.02 初期更敢探索
    vf_coef=cfg.ppo.vf_coef,
    n_steps=cfg.ppo.n_steps,
    batch_size=cfg.ppo.batch_size,
    n_epochs=cfg.ppo.n_epochs,
    target_kl=0.015,
    use_sde=True,
    sde_sample_freq=8,                              # 4~16 之间都行
    policy_kwargs=dict(
        net_arch=[128, 128],
        activation_fn=torch.nn.Tanh,
        ortho_init=True,
        optimizer_kwargs={"eps": 1e-5},
    ),
    max_grad_norm=0.5,
    device=cfg.device,
    seed=cfg.seed,
    tensorboard_log=os.path.join(cfg.save_path, "tb"),
    verbose=1
    )

    """eval_cb = EvalWithSuccessCallback(eval_env, eval_freq=max(cfg.env.eval_freq // cfg.n_envs, 1),
                                  n_eval_episodes=cfg.env.eval_episodes)"""
    eval_cb = EvalWithSuccessCallback(eval_env, milestones=(1000, 5000, 20000), n_eval_episodes=cfg.env.eval_episodes)

    metrics_cb = CustomMetricsCallback(
        noise_level=getattr(cfg.env, "noise_level", None),
        variant=getattr(cfg.env, "variant", None),
    )

    traj_cb = StateTrajectoryCallback(
        eval_env=eval_env,
        milestones=(1000, 5000, 20000),
        save_dir=os.path.join(cfg.save_path, "traj"),
        log_wandb=True,             # 你在用 wandb.init(sync_tensorboard=True) 时设 True
    )


    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[eval_cb, metrics_cb, traj_cb],
        progress_bar=True,
    )
    

    final_zip = os.path.join(cfg.save_path, f"{run_name}_final.zip")
    model.save(final_zip)

    # 保存 VecNormalize 统计
    vecnorm_pkl = os.path.join(cfg.save_path, "vecnorm.pkl")
    env.save(vecnorm_pkl)

    # 如果开了 track，则作为 artifact 上传
    if cfg.wandb.track:
        import wandb
        art = wandb.Artifact(f"{run_name}-artifacts", type="model")
        art.add_file(final_zip)
        art.add_file(vecnorm_pkl)
        wandb.log_artifact(art)

    env.close()
    eval_env.close()
    if cfg.wandb.track:
        wandb.finish() 


if __name__ == "__main__":
    main()
