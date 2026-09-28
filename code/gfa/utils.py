from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def joint_actions(action_sizes):
    if len(action_sizes) != 2:
        raise ValueError("This experiment code currently assumes two players.")
    return [(a0, a1) for a0 in range(action_sizes[0]) for a1 in range(action_sizes[1])]


def entropy(prob):
    prob = np.asarray(prob, dtype=float)
    mask = prob > 0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(prob[mask] * np.log(prob[mask])))


def soft_value(values, tau):
    values = np.asarray(values, dtype=float)
    if tau <= 0:
        return float(np.max(values))
    vmax = float(np.max(values))
    shifted = np.exp((values - vmax) / max(tau, 1e-12))
    return float(vmax + tau * np.log(np.sum(shifted)))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def clip_simplex(prob):
    prob = np.asarray(prob, dtype=float)
    prob = np.maximum(prob, 0.0)
    total = prob.sum()
    if total <= 0:
        return np.ones_like(prob) / len(prob)
    return prob / total


def l2_confidence_radius(dim, count, lam, scale):
    return scale * math.sqrt(dim / max(count, 1.0) + lam)
