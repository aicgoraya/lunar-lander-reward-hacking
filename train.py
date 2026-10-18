"""Train PPO on LunarLander-v3 with a chosen reward variant.

usage: python train.py <variant> [steps] [seed]
variants: baseline, timefee, hover, fix
"""
import sys
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from rewards import make_env

torch.set_num_threads(1)

variant = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
tag = f"{variant}_s{seed}".replace(":", "_")

env = make_vec_env(make_env(variant), n_envs=16, seed=seed,
                   monitor_dir=f"logs/{tag}")

# Hyperparameters from the Hugging Face Deep RL course, Unit 1
model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4,
            gamma=0.999, gae_lambda=0.98, ent_coef=0.01, seed=seed,
            device="cpu", verbose=1)

t0 = time.time()
model.learn(total_timesteps=steps)
print(f"TRAIN_TIME {tag} {time.time() - t0:.0f}s")
model.save(f"models/{tag}")
