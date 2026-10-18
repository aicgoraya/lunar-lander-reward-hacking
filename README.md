# lunar lander reward hacking

PPO on Gymnasium's LunarLander-v3, with a few changes to the reward to see how the agent's behavior changes.

<p>
<img src="media/landing.gif" width="300">
<img src="media/hover_hack.gif" width="300">
</p>

Left is the built-in reward. Right adds +1 for every frame the rocket is upright and moving slowly. With that bonus, hovering until the 1000 frame limit earns more than landing, so the agent hovers.

## results

Each model is scored on the normal game with the extra reward turned off. 1M training steps per run with Stable-Baselines3 PPO, about 5 minutes per run on CPU.

| reward | score | landed |
|---|---|---|
| built-in | 257 | 49/50 |
| built-in + small cost per frame | 267 | 50/50, ~15% faster |
| +1 per upright/slow frame | -89, -64 (2 seeds) | 0/40 |
| fixed version | 264, 271 (2 seeds) | 99/100 |

The fixed version pays for the change into the upright and slow state between frames, not for being in it. Over one episode that sums to at most +1, so hovering doesn't earn anything extra. It also keeps the per-frame cost.

![training curves](media/training_curves.png)

The hover agent's training reward reaches about 1000 even though it never lands.

## bonus size

Hover bonus swept from 0.02 to 1.0 per frame, 2 seeds each.

![bonus sweep](media/bonus_sweep.png)

The agent still lands at +0.5, but landings take longer from +0.2 on (about 40% longer). By +1.0 it never lands. At +0.7 the two seeds landed 43% and 87% of the time.

Raw numbers for every run are in `results/`.

## run it

```
pip install swig
pip install "gymnasium[box2d]" stable-baselines3 imageio imageio-ffmpeg pandas matplotlib
python train.py baseline          # or timefee, hover, fix, hover:0.5
python evaluate.py baseline_s0    # scores on the normal game and saves a clip
```

All reward versions are in `rewards.py`.
