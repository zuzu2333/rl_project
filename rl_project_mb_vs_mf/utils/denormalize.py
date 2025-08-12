import pickle
import numpy as np

# 你的 vecnorm.pkl 路径
vecnorm_path = "vecnorm.pkl"

with open(vecnorm_path, "rb") as f:
    vecnorm_data = pickle.load(f)

obs_rms = vecnorm_data.obs_rms

# 均值和标准差
obs_mean = obs_rms.mean
obs_std = np.sqrt(obs_rms.var)

print("obs_mean:", obs_mean)
print("obs_std:", obs_std)
