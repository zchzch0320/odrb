from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .utils import clip_simplex, entropy, joint_actions


def max_cce_violation(payoffs, dist, action_sizes):
    n_players = payoffs.shape[0]
    actions = joint_actions(action_sizes)
    dist = np.asarray(dist, dtype=float)
    total = 0.0
    per_player = []
    for i in range(n_players):
        current = float(np.dot(dist, payoffs[i]))
        worst = -np.inf
        for alt in range(action_sizes[i]):
            alt_payoff = 0.0
            for idx, action in enumerate(actions):
                alt_action = list(action)
                alt_action[i] = alt
                alt_idx = actions.index(tuple(alt_action))
                alt_payoff += dist[idx] * payoffs[i, alt_idx]
            worst = max(worst, alt_payoff - current)
        per_player.append(float(worst))
        total = max(total, worst)
    return float(total), per_player


def _is_zero_sum(payoffs):
    payoffs = np.asarray(payoffs, dtype=float)
    return payoffs.shape[0] == 2 and np.allclose(payoffs[0] + payoffs[1], 0.0, atol=1e-8)


def _solve_zero_sum_stage(payoffs, action_sizes, tau=0.0):
    row_actions, col_actions = action_sizes
    matrix = np.asarray(payoffs[0], dtype=float).reshape(row_actions, col_actions)

    row_c = np.zeros(row_actions + 1, dtype=float)
    row_c[-1] = -1.0
    row_a_ub = np.hstack([-matrix.T, np.ones((col_actions, 1), dtype=float)])
    row_b_ub = np.zeros(col_actions, dtype=float)
    row_a_eq = np.zeros((1, row_actions + 1), dtype=float)
    row_a_eq[0, :row_actions] = 1.0
    row_b_eq = np.asarray([1.0], dtype=float)
    row_bounds = [(0.0, 1.0) for _ in range(row_actions)] + [(None, None)]
    row_res = linprog(
        c=row_c,
        A_ub=row_a_ub,
        b_ub=row_b_ub,
        A_eq=row_a_eq,
        b_eq=row_b_eq,
        bounds=row_bounds,
        method="highs",
    )

    col_c = np.zeros(col_actions + 1, dtype=float)
    col_c[-1] = 1.0
    col_a_ub = np.hstack([matrix, -np.ones((row_actions, 1), dtype=float)])
    col_b_ub = np.zeros(row_actions, dtype=float)
    col_a_eq = np.zeros((1, col_actions + 1), dtype=float)
    col_a_eq[0, :col_actions] = 1.0
    col_b_eq = np.asarray([1.0], dtype=float)
    col_bounds = [(0.0, 1.0) for _ in range(col_actions)] + [(None, None)]
    col_res = linprog(
        c=col_c,
        A_ub=col_a_ub,
        b_ub=col_b_ub,
        A_eq=col_a_eq,
        b_eq=col_b_eq,
        bounds=col_bounds,
        method="highs",
    )

    if row_res.success:
        row_mix = clip_simplex(row_res.x[:row_actions])
    else:
        row_mix = np.ones(row_actions, dtype=float) / row_actions
    if col_res.success:
        col_mix = clip_simplex(col_res.x[:col_actions])
    else:
        col_mix = np.ones(col_actions, dtype=float) / col_actions

    if tau > 0:
        row_uniform = np.ones(row_actions, dtype=float) / row_actions
        col_uniform = np.ones(col_actions, dtype=float) / col_actions
        mix = min(0.35, 5.0 * tau)
        row_mix = clip_simplex((1.0 - mix) * row_mix + mix * row_uniform)
        col_mix = clip_simplex((1.0 - mix) * col_mix + mix * col_uniform)

    return clip_simplex(np.outer(row_mix, col_mix).reshape(-1))


def solve_stage_cce(payoffs, action_sizes, tau=0.0, approx_tol=1e-6, violation_scale=0.1, refine_steps=24):
    payoffs = np.asarray(payoffs, dtype=float)
    if _is_zero_sum(payoffs):
        return _solve_zero_sum_stage(payoffs, action_sizes, tau=tau)
    n_players, joint_count = payoffs.shape
    actions = joint_actions(action_sizes)
    welfare = payoffs.sum(axis=0)
    a_ub = []
    b_ub = []
    for i in range(n_players):
        for alt in range(action_sizes[i]):
            coeff = np.zeros(joint_count, dtype=float)
            for idx, action in enumerate(actions):
                alt_action = list(action)
                alt_action[i] = alt
                alt_idx = actions.index(tuple(alt_action))
                coeff[idx] = -(payoffs[i, idx] - payoffs[i, alt_idx])
            a_ub.append(coeff)
            b_ub.append(approx_tol)
    res = linprog(
        c=-welfare,
        A_ub=np.asarray(a_ub, dtype=float) if a_ub else None,
        b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
        A_eq=np.asarray([np.ones(joint_count, dtype=float)]),
        b_eq=np.asarray([1.0], dtype=float),
        bounds=[(0.0, 1.0) for _ in range(joint_count)],
        method="highs",
    )
    if not res.success:
        return np.ones(joint_count, dtype=float) / joint_count
    base = clip_simplex(res.x)
    if tau <= 0:
        return base
    uniform = np.ones(joint_count, dtype=float) / joint_count
    lo = 0.0
    hi = min(0.5, 5.0 * tau)
    best = base
    best_entropy = entropy(base)
    for _ in range(refine_steps):
        mid = 0.5 * (lo + hi)
        cand = clip_simplex((1.0 - mid) * base + mid * uniform)
        violation, _ = max_cce_violation(payoffs, cand, action_sizes)
        if violation <= max(approx_tol, violation_scale * tau):
            if entropy(cand) >= best_entropy:
                best = cand
                best_entropy = entropy(cand)
            lo = mid
        else:
            hi = mid
    return best


def discretize_stage_game(payoffs, resolution):
    payoffs = np.asarray(payoffs, dtype=float)
    if resolution is None or resolution <= 0:
        return np.array(payoffs, copy=True)
    return resolution * np.round(payoffs / resolution)


def marginals_from_joint(dist, action_sizes):
    dist = np.asarray(dist, dtype=float)
    acts = joint_actions(action_sizes)
    marginals = []
    for i, size in enumerate(action_sizes):
        marginal = np.zeros(size, dtype=float)
        for idx, action in enumerate(acts):
            marginal[action[i]] += dist[idx]
        marginals.append(clip_simplex(marginal))
    return marginals
