"""Evaluate a trained model on the PLAIN game (no custom reward) and record a clip.

usage: python evaluate.py <tag> [episodes]
Prints: mean built-in score, landing rate, crash rate, timeouts, mean episode length.
Writes videos/<tag>.mp4 (first 3 episodes).
"""
import json
import sys

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO

tag = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 50

model = PPO.load(f"models/{tag}", device="cpu")
env = gym.make("LunarLander-v3", render_mode="rgb_array")

scores, lengths, outcomes, frames = [], [], [], []
for ep in range(n):
    obs, _ = env.reset(seed=10_000 + ep)
    done, total, t = False, 0.0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        total += r
        t += 1
        done = term or trunc
        if ep < 3:
            frames.append(env.render())
    # On termination the last reward is +100 (rested) or -100 (crash/out of bounds)
    outcomes.append("landed" if term and r > 0 else "crashed" if term else "timeout")
    scores.append(total)
    lengths.append(t)

scores = np.array(scores)
res = {
    "tag": tag,
    "episodes": n,
    "mean_score": round(float(scores.mean()), 1),
    "std_score": round(float(scores.std()), 1),
    "solved_rate": round(float((scores >= 200).mean()), 2),
    "landed": outcomes.count("landed"),
    "crashed": outcomes.count("crashed"),
    "timeout": outcomes.count("timeout"),
    "mean_length": round(float(np.mean(lengths)), 0),
}
print(json.dumps(res))
with open(f"results/{tag}.json", "w") as f:
    json.dump(res, f)
imageio.mimsave(f"videos/{tag}.mp4", frames, fps=50, macro_block_size=1)
