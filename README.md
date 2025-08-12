# rl_project
## 🔄 Reproducibility Checklist

To reproduce the results in this repository, please ensure the following:

- **Environment**
  - [ ] OS and version: e.g. Ubuntu 20.04 / Windows 11 / macOS 13
  - [ ] Python version: `3.11` (tested with Python Vision)
  - [ ] Virtual environment: [uv](https://github.com/astral-sh/uv)

- **Code & Config**
  - [ ] Dependencies listed in `pyproject.toml`
  - [ ] Install via:
    ```bash
    make install
    ```
  - [ ] Use provided config files in `configs/` for each experiment
  - [ ] Ensure **random seeds** are set consistently (default: 10 seeds per experiment)
  - [ ] Algorithms:
    - PPO implementation from Stable-Baselines3
    - MPC implementation from `mb_agent.py`
  - [ ] Training scripts:
    - PPO:
      ```bash
      python train_ppo.py -m seed=0,1,...
      ```
    - MPC:
      ```bash
      python train_mpc.py -m seed=0,1,...
      ```

- **Data & Logging**
  - [ ] All logs and models stored in W&B project: [wandb.ai/jingtaoz/rl_fp](https://wandb.ai/jingtaoz/rl_fp?nw=nwuserjingtaozhu)
  - [ ] Logging includes training curves, metrics, and checkpoints

- **Experiments**
  - [ ] Environment:
    - `MountainCarContinuous-v0` (Gymnasium)
  - [ ] Budgets: `1k`, `5k`, `20k` steps
