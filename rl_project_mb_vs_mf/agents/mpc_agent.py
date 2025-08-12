# mpc_agent.py
import os, time, random
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from omegaconf import DictConfig, OmegaConf
import hydra

class PlannerCtx:
    def __init__(self, act_low, act_high, horizon, num_candidates, gamma, device, action_std=None):
        self.device = device
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.action_std = action_std

        self.low  = torch.tensor(act_low,  dtype=torch.float32, device=device)
        self.high = torch.tensor(act_high, dtype=torch.float32, device=device)
        self.mean = (self.low + self.high) / 2.0
        self.std  = (self.high - self.low) * (action_std if action_std is not None else 1.0)

        act_dim = self.low.shape[0]
        self.actions = torch.empty(num_candidates, horizon, act_dim, device=device)
        self.discount = (gamma ** torch.arange(horizon, device=device)).to(torch.float32)

    @torch.no_grad()
    def plan(self, model, state_np):
        model.eval()
        s0 = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 采样候选：高斯或均匀（就地）
        if self.action_std is None:
            self.actions.uniform_(0.0, 1.0)
            self.actions.mul_(self.high - self.low).add_(self.low)
        else:
            self.actions.normal_()
            self.actions.mul_(self.std).add_(self.mean).clamp_(self.low, self.high)

        # 批量 rollout：每个 t 一个前向，但并行 K 条轨迹
        states = s0.repeat(self.num_candidates, 1)
        total = torch.zeros(self.num_candidates, device=self.device)
        for t in range(self.horizon):
            a_t = self.actions[:, t, :]
            s_next = model.next_state(states, a_t)
            # 启发式 proxy 奖励（与原版一致）
            pos = s_next[:, 0]
            proxy_r = -(0.45 - pos).abs()
            total.add_(proxy_r * self.discount[t])
            states = s_next

        best = torch.argmax(total).item()
        a0 = self.actions[best, 0, :].detach().cpu().numpy()
        return np.clip(a0, self.low.cpu().numpy(), self.high.cpu().numpy())


def eval_policy_with_success(env, planner_fn, n_episodes=5, goal_fallback=0.45, log_traj=False, tag=None, media_fn=None):
    """env 为单环境 Gymnasium，planner_fn(state)->action"""
    ep_rewards, ep_lengths, successes = [], [], []
    traj_png = None
    for ep in range(n_episodes):
        s, _ = env.reset()
        done = truncated = False
        total_r, length = 0.0, 0
        pos_hist, vel_hist, act_hist, t_hist = [], [], [], []
        while not (done or truncated):
            a = planner_fn(s)
            s2, r, done, truncated, info = env.step(a)
            total_r += r; length += 1; s = s2
            # 记录状态
            pos_hist.append(float(s[0])); vel_hist.append(float(s[1])); act_hist.append(float(a[0] if np.ndim(a) else a)); t_hist.append(length)

        # success 判定（MountainCar）
        try:
            goal = getattr(env.unwrapped, "goal_position", goal_fallback)
        except Exception:
            goal = goal_fallback
        success = 1.0 if float(s[0]) >= float(goal) else 0.0

        ep_rewards.append(total_r); ep_lengths.append(length); successes.append(success)

        # 只在里程碑时画一条
        if log_traj and media_fn is not None and ep == 0 and tag is not None:
            import matplotlib.pyplot as plt, os
            os.makedirs("./artifacts/traj", exist_ok=True)
            png_path = f"./artifacts/traj/{tag}.png"
            plt.figure(figsize=(8,4))
            plt.plot(t_hist, pos_hist, label="position")
            plt.plot(t_hist, vel_hist, label="velocity")
            plt.xlabel("time step"); plt.ylabel("state"); plt.title(tag); plt.legend(); plt.tight_layout()
            plt.savefig(png_path, dpi=150); plt.close()
            try:
                import wandb
                if wandb.run is not None:
                    media_fn({"trajectory/image": wandb.Image(png_path), "trajectory/tag": tag})
            except Exception:
                pass
            traj_png = png_path

    metrics = {
        "eval/mean_reward": float(np.mean(ep_rewards)),
        "eval/mean_ep_length": float(np.mean(ep_lengths)),
        "eval/success_rate": float(np.mean(successes)),
    }
    return metrics, traj_png


class MPCMetrics:
    def __init__(self, ema_alpha=0.1, hist_window_episodes=100):
        # runtime
        self._ema_plan = None
        self._last_wall_time = None
        self._last_steps_for_wall = 0
        self._last_sps_time = None
        self._last_sps_steps = 0
        # episodic hists
        self.rew_hist = deque(maxlen=hist_window_episodes)
        self.len_hist = deque(maxlen=hist_window_episodes)
        self.cur_ep_ret = 0.0
        self.cur_ep_len = 0

    def on_training_start(self, cur_steps):
        now = time.time()
        self._last_wall_time = now
        self._last_steps_for_wall = cur_steps
        self._last_sps_time = now
        self._last_sps_steps = cur_steps

    def on_step(self, planning_dt, cur_steps, log_fn):
        # planning time (EMA)
        alpha = 0.1
        self._ema_plan = planning_dt if self._ema_plan is None else (1 - alpha) * self._ema_plan + alpha * planning_dt
        log_fn({"perf/planning_time_per_step": float(self._ema_plan)})

        # SPS
        now = time.time()
        sps_dt = now - self._last_sps_time
        sps_dsteps = cur_steps - self._last_sps_steps
        if sps_dt > 0:
            log_fn({"charts/SPS": float(sps_dsteps / sps_dt)})
            self._last_sps_time = now
            self._last_sps_steps = cur_steps

        # wallclock per 1k steps
        dsteps = cur_steps - self._last_steps_for_wall
        if dsteps >= 1000:
            dt = now - self._last_wall_time
            if dt > 0:
                log_fn({"time/wallclock_per_1k_steps": float(dt * (1000.0 / dsteps))})
            self._last_wall_time = now
            self._last_steps_for_wall = cur_steps

    def on_env_step(self, reward):
        self.cur_ep_ret += float(reward)
        self.cur_ep_len += 1

    def on_episode_end(self, log_fn):
        self.rew_hist.append(self.cur_ep_ret)
        self.len_hist.append(self.cur_ep_len)
        # 原始分布（对齐 PPO 的 charts/* 命名）
        if len(self.rew_hist) == self.rew_hist.maxlen:
            r = np.asarray(self.rew_hist, np.float32)
            l = np.asarray(self.len_hist, np.float32)
            for name, arr in [("charts/episodic_return_raw", r), ("charts/episodic_length_raw", l)]:
                q10, q50, q90 = np.percentile(arr, [10, 50, 90])
                log_fn({
                    name: arr, f"{name}_p10": float(q10),
                    f"{name}_p50": float(q50), f"{name}_p90": float(q90),
                    f"{name}_std": float(arr.std(ddof=0)),
                })
        self.cur_ep_ret, self.cur_ep_len = 0.0, 0


# ===== 简单 MLP 模型：输入 [s, a]，输出 Δs =====
class DynamicsModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128,128), predict_delta=True):
        super().__init__()
        self.predict_delta = predict_delta
        layers = []
        last = obs_dim + act_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, obs_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        out = self.net(x)
        return out  # Δs 或 直接 s'

    def next_state(self, s, a):
        pred = self.forward(s, a)
        return s + pred if self.predict_delta else pred


# ===== 简单经验回放 =====
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=100000):
        self.size = size
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.nxt = np.zeros((size, obs_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.ptr = 0; self.full = False

    def add(self, s, a, s2, r, d):
        self.obs[self.ptr] = s; self.act[self.ptr] = a
        self.nxt[self.ptr] = s2; self.rew[self.ptr] = r; self.done[self.ptr] = d
        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0; self.full = True

    def sample(self, batch):
        max_n = self.size if self.full else self.ptr
        idx = np.random.randint(0, max_n, size=batch)
        return ( self.obs[idx], self.act[idx], self.nxt[idx], self.rew[idx], self.done[idx] )


# ===== 规划：Random Shooting =====
@torch.no_grad()
def plan_action(model, state, action_low, action_high, horizon, num_candidates, gamma, device, action_std=None):
    """
    state: np.array (obs_dim,)
    返回：最佳序列的第一个动作
    """
    obs_dim = state.shape[0]
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1,obs)

    # 采样动作序列 (num_candidates, horizon, act_dim)
    act_dim = action_low.shape[0]
    low = torch.tensor(action_low, device=device)
    high = torch.tensor(action_high, device=device)

    if action_std is None:
        # 均匀采样
        actions = low + torch.rand(num_candidates, horizon, act_dim, device=device) * (high - low)
    else:
        # 高斯采样 + clip
        mean = (low + high) / 2.0
        std = (high - low) * action_std
        actions = torch.randn(num_candidates, horizon, act_dim, device=device) * std + mean
        actions = torch.max(torch.min(actions, high), low)

    # 复制初始状态
    states = state_t.repeat(num_candidates, 1)  # (N, obs)
    total_return = torch.zeros(num_candidates, device=device)

    discount = 1.0
    for t in range(horizon):
        a_t = actions[:, t, :]
        s_next = model.next_state(states, a_t)
        # 即时 reward：对 MountainCarContinuous，环境内部计算较复杂；
        # 这里用一个近似 shaped-reward：负的能量消耗 (鼓励小油门) + 到达目标大回报可由真实环境给出。
        # 训练/评估时，我们仍然依赖真实环境给 reward；规划时仅用“启发式”累计距离目标的负势能。
        # 为简单：用位置越接近目标的“负距离”作为 proxy（只用于排序）。
        pos = s_next[:, 0]
        vel = s_next[:, 1]
        # 终点近似：> 0.45 视为到旗子（MCC 的 goal_position）
        goal = 0.45

        # 组合 proxy：鼓励速度、惩罚过大油门（更稳）
        energy_penalty = (a_t.square().sum(dim=1))    # ||a||^2
        proxy_r = - (goal - pos).abs() + 0.08 * vel - 0.001 * energy_penalty

        # 终点 bonus（仅用于排序，避免“刚到就掉头”）
        proxy_r = torch.where(pos >= goal, proxy_r + 5.0, proxy_r)    # 目标位置 ~0.45（接近旗子），只是启发式
        total_return += discount * proxy_r
        states = s_next
        discount *= gamma

    # 取分数最高的序列第一个动作
    best_idx = torch.argmax(total_return).item()
    best_a0 = actions[best_idx, 0, :].cpu().numpy()
    return np.clip(best_a0, action_low, action_high)


# ========= 在 train_model 里补充：返回 train_loss 和 val_loss =========
def train_model(model, buffer, cfg, device):
    opt = optim.Adam(model.parameters(), lr=cfg.mpc.lr, weight_decay=cfg.mpc.weight_decay)
    loss_fn = nn.MSELoss()
    model.train()
    total = 0.0
    for _ in range(cfg.mpc.gradient_steps):
        s, a, s2, _, _ = buffer.sample(cfg.mpc.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=device)
        a = torch.tensor(a, dtype=torch.float32, device=device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=device)
        pred_next = model.next_state(s, a)
        loss = loss_fn(pred_next, s2)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    # 计算一个轻量 val_loss（随机采样）
    with torch.no_grad():
        s, a, s2, _, _ = buffer.sample(min(2048, cfg.mpc.batch_size*2))
        s = torch.tensor(s, dtype=torch.float32, device=device)
        a = torch.tensor(a, dtype=torch.float32, device=device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=device)
        pred_next = model.next_state(s, a)
        val_loss = float(nn.MSELoss()(pred_next, s2).item())
    return total / max(1, cfg.mpc.gradient_steps), val_loss

# ========= 在主循环外新增：保存最近一次完整轨迹（用于 rollout MSE）=========
class RecentTrajectory:
    def __init__(self, max_len=2000):
        self.obs, self.act, self.next_obs = [], [], []
        self.max_len = max_len
    def add(self, s, a, s2):
        self.obs.append(s.copy()); self.act.append(a.copy()); self.next_obs.append(s2.copy())
        if len(self.obs) > self.max_len:
            self.obs.pop(0); self.act.pop(0); self.next_obs.pop(0)
    def sample_episode_tail(self, length=300):
        L = min(length, len(self.obs))
        if L < 3: return None
        obs = np.array(self.obs[-L:], dtype=np.float32)
        act = np.array(self.act[-L:], dtype=np.float32)
        nxt = np.array(self.next_obs[-L:], dtype=np.float32)
        return obs, act, nxt

def rollout_mse(model, traj, Ks=(5,20,50), device=None):
    """基于最近一段真实 (s_t, a_t, s_{t+1}) 计算开环 k-step 误差"""
    pack = traj.sample_episode_tail(length=500)
    if pack is None: 
        return {}
    s, a, s_next = pack
    out = {}
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        s_t = torch.tensor(s[0], dtype=torch.float32, device=device).unsqueeze(0)
        t_idx = 0
        for k in sorted(Ks):
            s_pred = s_t.clone()
            for i in range(k):
                if t_idx+i >= len(a): break
                a_i = torch.tensor(a[t_idx+i], dtype=torch.float32, device=device).unsqueeze(0)
                s_pred = model.next_state(s_pred, a_i)
            if t_idx+k < len(s_next):
                gt = torch.tensor(s[t_idx+k], dtype=torch.float32, device=device).unsqueeze(0)
                out[f"model/rollout_MSE@{k}"] = float(nn.MSELoss()(s_pred, gt).item())
    return out



def evaluate(env, model, cfg, device, action_low, action_high):
    ep_rewards = []
    for _ in range(cfg.env.eval_episodes):
        s, _ = env.reset()
        done, truncated = False, False
        total_r = 0.0
        while not (done or truncated):
            a = plan_action(
                model, s, action_low, action_high,
                horizon=cfg.mpc.planning_horizon,
                num_candidates=cfg.mpc.num_candidates,
                gamma=cfg.mpc.gamma,
                device=device,
                action_std=cfg.mpc.action_sample_std
            )
            s, r, done, truncated, _ = env.step(a)
            total_r += r
        ep_rewards.append(total_r)
    return float(np.mean(ep_rewards))


from stable_baselines3.common.logger import configure

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Final config:\n", OmegaConf.to_yaml(cfg))

    run_name = f"{cfg.env.env_id}_mpc_s{cfg.seed}_{int(time.time())}"
    os.makedirs(cfg.save_path, exist_ok=True)

    # --- device
    device = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else cfg.device)
    if cfg.device != "auto":
        device = torch.device(cfg.device)

    # --- W&B（只用于同步TB & 记录媒体）
    if cfg.wandb.track:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            sync_tensorboard=True,           # ★ 同步 SB3 的 TensorBoard
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name, group=cfg.wandb.group, tags=cfg.wandb.tags,
            monitor_gym=True, save_code=True,
        )

    # --- SB3 logger：stdout + tensorboard（可加 "csv"）
    tb_dir = os.path.join(cfg.save_path, "tb")
    logger = configure(folder=tb_dir, format_strings=["stdout", "tensorboard"])

    # —— 工具：媒体打点（仅图片/表格）
    def log_media(step: int, data: dict):
        if cfg.wandb.track:
            import wandb
            if wandb.run is not None:
                wandb.log(data, step=step)

    # --- Env
    env = gym.make(cfg.env.env_id)
    env.action_space.seed(cfg.seed)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    EVAL_STRIDE = 1_000
    MIN_EVAL_INTERVAL_SEC = getattr(cfg.env, "min_eval_interval_sec", 0)
    next_eval_step = 0
    last_eval_wall = time.time() - MIN_EVAL_INTERVAL_SEC   # 允许首评立即触发

    # --- Buffer & Model
    buffer = ReplayBuffer(obs_dim, act_dim, size=cfg.mpc.buffer_size)
    model = DynamicsModel(obs_dim, act_dim, tuple(cfg.mpc.hidden_sizes),
                          predict_delta=cfg.mpc.predict_delta).to(device)
    traj = RecentTrajectory(max_len=3000)
    
    planner = PlannerCtx(
        act_low=act_low,
        act_high=act_high,
        horizon=cfg.mpc.planning_horizon,
        num_candidates=cfg.mpc.num_candidates,
        gamma=cfg.mpc.gamma,
        device=device,
        action_std=cfg.mpc.action_sample_std,
    )

    # ===== 规划参数/稳定性（一次性记录）=====
    logger.record("plan/horizon_H",   cfg.mpc.planning_horizon)
    logger.record("plan/candidates_N", cfg.mpc.num_candidates)
    logger.record("plan/noise_std",   cfg.mpc.action_sample_std)
    logger.record("plan/gamma",       cfg.mpc.gamma)
    logger.record("plan/warm_start",  float(cfg.mpc.warmup_steps > 0))
    logger.dump(0)  # 作为 step=0 的初始状态

    # ===== 运行时性能度量器 =====
    metrics = MPCMetrics()
    metrics.on_training_start(cur_steps=0)

    total_steps = 0
    episode = 0
    s, _ = env.reset()

    while total_steps < cfg.total_timesteps:
        # —— 动作选择
        t0 = time.perf_counter()
        if total_steps < cfg.mpc.warmup_steps:
            a = env.action_space.sample()
        else:
            a = planner.plan(model, s)
        planning_dt = time.perf_counter() - t0

        # —— 环境一步
        s2, r, done, truncated, _ = env.step(a)
        buffer.add(s, a, s2, r, float(done or truncated))
        traj.add(s, a, s2)
        s = s2
        total_steps += 1

        # —— 运行时性能（交给 SB3 logger 记标量）
        def _log_runtime(d: dict):
            for k, v in d.items():
                logger.record(k, float(v))
        metrics.on_step(planning_dt, total_steps, _log_runtime)
        metrics.on_env_step(r)

        # —— 回合结束
        if done or truncated:
            episode += 1
            def _log_ep(d: dict):
                # d 里有 charts/episodic_*_raw 的分位数/方差标量，逐项记录
                for k, v in d.items():
                    # 原始数组不写入 TB（如需存表/图片再用 wandb）
                    if isinstance(v, (float, int)):
                        logger.record(k, float(v))
            metrics.on_episode_end(_log_ep)
            s, _ = env.reset()

        # —— 训练 dynamics 模型
        if total_steps >= cfg.mpc.warmup_steps and total_steps % cfg.mpc.train_freq == 0:
            train_loss, val_loss = train_model(model, buffer, cfg, device)
            logger.record("model/train_loss",  float(train_loss))
            logger.record("model/val_loss",    float(val_loss))
            logger.record("model/one_step_MSE", float(val_loss))

            # 多步开环误差（标量）
            mseks = rollout_mse(model, traj, Ks=(5,10,20), device=device)  # PATCH: 用真实收集的 traj
            for k, v in mseks.items():
                logger.record(k, float(v))


            # （可选）如果你算了 planning 平均耗时/SPS，请在 metrics 里返回并 record 这里
            logger.record("time/steps",        total_steps)
            logger.dump(total_steps)  # ★ 刷到 stdout/TB，并提供 global step
        

        need_eval = (total_steps >= next_eval_step) and \
            ((time.time() - last_eval_wall) >= MIN_EVAL_INTERVAL_SEC)

        if need_eval:
            # 里程碑时多评几条，其它时候评少一点
            n_eval_eps = cfg.env.eval_episodes
            if total_steps not in (10_000, 20_000, 0):
                n_eval_eps = max(2, min(3, n_eval_eps))

            eval_metrics, _ = eval_policy_with_success(
                env,
                planner_fn=lambda st: plan_action(
                    model, st, act_low, act_high,
                    cfg.mpc.planning_horizon, cfg.mpc.num_candidates,
                    cfg.mpc.gamma, device, cfg.mpc.action_sample_std
                ),
                n_episodes=n_eval_eps,
                log_traj=True,
                tag=f"traj_step_{total_steps}",
                media_fn=lambda d: log_media(total_steps, d)
            )
            for k, v in eval_metrics.items():
                logger.record(k, float(v))
            logger.record("time/steps", total_steps)
            logger.dump(total_steps)

            # 固定步距 + 更新墙钟
            next_eval_step = total_steps + EVAL_STRIDE
            last_eval_wall = time.time()

        # —— 周期性 flush，保证 runtime 指标按 env_steps 刷新
        if total_steps % getattr(cfg, "log_flush_freq", 200) == 0:
            logger.record("time/steps", total_steps)
            logger.dump(total_steps)
    
    eval_metrics, _ = eval_policy_with_success(
        env,
        planner_fn=lambda st: plan_action(
            model, st, act_low, act_high,
            cfg.mpc.planning_horizon, cfg.mpc.num_candidates,
            cfg.mpc.gamma, device, cfg.mpc.action_sample_std
        ),
        n_episodes=cfg.env.eval_episodes,
        log_traj=True,
        tag=f"traj_step_{total_steps}",
        media_fn=lambda d: log_media(total_steps, d)
    )
    for k, v in eval_metrics.items():
        logger.record(k, float(v))
    logger.record("time/steps", total_steps)
    logger.dump(total_steps)

    # --- 保存
    os.makedirs(os.path.dirname(cfg.mpc.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), cfg.mpc.model_save_path)
    logger.record("artifacts/model_path", 1.0)  # 仅做标记
    logger.dump(total_steps)

    if cfg.wandb.track:
        import wandb
        wandb.save(cfg.mpc.model_save_path)
        wandb.finish()



if __name__ == "__main__":
    main()
