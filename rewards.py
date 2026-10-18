"""Reward-shaping wrappers for LunarLander-v3.

Observation layout: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
"""
import gymnasium as gym
import numpy as np


def upright_and_slow(obs, max_tilt=0.2, max_speed=0.3):
    speed = float(np.hypot(obs[2], obs[3]))
    return abs(obs[4]) < max_tilt and speed < max_speed


class TimeFee(gym.Wrapper):
    """Built-in reward minus a small charge every frame (built-in has none)."""

    def __init__(self, env, fee=0.05):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info["base_reward"] = r
        return obs, r - self.fee, term, trunc, info


class HoverHack(gym.Wrapper):
    """The broken rule: +1 for every frame the rocket is upright and slow.

    Pays for *being* in a nice state, so hovering for 1,000 frames beats landing.
    """

    def __init__(self, env, bonus=1.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info["base_reward"] = r
        if upright_and_slow(obs):
            r += self.bonus
        return obs, r, term, trunc, info


class HoverFix(gym.Wrapper):
    """The fix: pay for *getting* upright and slow (change between frames),
    not for staying there, plus a per-frame fee so waiting costs something.

    bonus_t = scale * (phi(s_t+1) - phi(s_t)), phi = 1 if upright and slow.
    Summed over an episode this telescopes to at most `scale`, so hovering
    can't farm it. On the terminal frame phi(s_T+1) is treated as 0... except
    we keep it simple and only telescope within the episode.
    """

    def __init__(self, env, scale=1.0, fee=0.05):
        super().__init__(env)
        self.scale = scale
        self.fee = fee
        self.prev_phi = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_phi = float(upright_and_slow(obs))
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info["base_reward"] = r
        phi = float(upright_and_slow(obs))
        r += self.scale * (phi - self.prev_phi) - self.fee
        self.prev_phi = phi
        return obs, r, term, trunc, info


WRAPPERS = {
    "baseline": None,
    "timefee": TimeFee,
    "hover": HoverHack,
    "fix": HoverFix,
}


def make_env(variant, render_mode=None):
    """variant is a key of WRAPPERS, or 'hover:<bonus>' for a sized hover bonus."""
    def _f():
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        if variant.startswith("hover:"):
            return HoverHack(env, bonus=float(variant.split(":")[1]))
        w = WRAPPERS[variant]
        return w(env) if w else env
    return _f
