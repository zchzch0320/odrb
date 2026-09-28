from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np

from .solvers import discretize_stage_game, marginals_from_joint, solve_stage_cce
from .utils import clip_simplex, entropy, joint_actions, set_seed, soft_value


@dataclass
class AlgoConfig:
    name: str
    robust: str
    beta: float
    tau: float
    linear: bool = False
    update_every: int = 10
    lam: float = 1.0
    bonus_scale: float = 1.0
    eta_scale: float = 0.0
    xi_scale: float = 1e-6
    stage_violation_scale: float = 0.1
    stage_refine_steps: int = 24


def make_algo_config(
    name,
    beta_override=None,
    tau_override=None,
    lam_override=None,
    eta_scale_override=None,
    xi_scale_override=None,
    stage_violation_scale_override=None,
    stage_refine_steps_override=None,
    update_every_override=None,
):
    mapping = {
        "nonrobust_cce": AlgoConfig("NonRobust-CCE", "none", 0.0, 0.0, linear=False),
        "policyreg_only": AlgoConfig("PolicyReg-only", "none", 0.0, 0.02, linear=False),
        "robust_tv": AlgoConfig("Robust-TV", "tv", 0.20, 0.0, linear=False),
        "robust_kl": AlgoConfig("Robust-KL", "kl", 0.20, 0.0, linear=False),
        "robust_chi2": AlgoConfig("Robust-$\\chi^2$", "chi2", 0.20, 0.0, linear=False),
        "drmg_tv": AlgoConfig("DRMG-TV", "tv", 0.20, 0.02, linear=False),
        "drmg_kl": AlgoConfig("DRMG-KL", "kl", 0.20, 0.02, linear=False),
        "drmg_chi2": AlgoConfig("DRMG-$\\chi^2$", "chi2", 0.20, 0.02, linear=False),
        "nqovi_l": AlgoConfig("NQOVI-L", "none", 0.0, 0.0, linear=True),
        "policyreg_l": AlgoConfig("PolicyReg-L", "none", 0.0, 0.02, linear=True),
        "dr_cce_lsi": AlgoConfig("DR-CCE-LSI", "tv", 0.20, 0.0, linear=True),
        "dr_cce_lsi_kl": AlgoConfig("DR-CCE-LSI-KL", "kl", 0.20, 0.0, linear=True),
        "dr_cce_lsi_chi2": AlgoConfig("DR-CCE-LSI-$\\chi^2$", "chi2", 0.20, 0.0, linear=True),
        "drmg_tv_l": AlgoConfig("DRMG-TV-L", "tv", 0.20, 0.02, linear=True),
        "drmg_kl_l": AlgoConfig("DRMG-KL-L", "kl", 0.20, 0.02, linear=True),
        "drmg_chi2_l": AlgoConfig("DRMG-$\\chi^2$-L", "chi2", 0.20, 0.02, linear=True),
        "route_a_tabular": AlgoConfig("Route-A-Tabular", "tv", 0.20, 0.02, linear=False),
        "sigma0_tabular": AlgoConfig("Sigma0-Tabular", "none", 0.0, 0.02, linear=False),
        "no_exploiter_tabular": AlgoConfig("No-Exploiter-Tabular", "tv", 0.20, 0.0, linear=False),
        "route_a_linear": AlgoConfig("Route-A-Linear", "tv", 0.20, 0.02, linear=True),
        "sigma0_linear": AlgoConfig("Sigma0-Linear", "none", 0.0, 0.02, linear=True),
        "no_exploiter_linear": AlgoConfig("No-Exploiter-Linear", "tv", 0.20, 0.0, linear=True),
    }
    if name not in mapping:
        raise ValueError(f"Unknown algorithm: {name}")
    algo = mapping[name]
    if beta_override is not None:
        algo.beta = float(beta_override)
    if tau_override is not None:
        algo.tau = float(tau_override)
    if lam_override is not None:
        algo.lam = float(lam_override)
    if eta_scale_override is not None:
        algo.eta_scale = float(eta_scale_override)
    if xi_scale_override is not None:
        algo.xi_scale = float(xi_scale_override)
    if stage_violation_scale_override is not None:
        algo.stage_violation_scale = float(stage_violation_scale_override)
    if stage_refine_steps_override is not None:
        algo.stage_refine_steps = int(stage_refine_steps_override)
    if update_every_override is not None:
        algo.update_every = int(update_every_override)
    return algo


def robust_value(robust, beta, probs, values):
    probs = np.asarray(probs, dtype=float)
    values = np.asarray(values, dtype=float)
    mean = float(np.dot(probs, values))
    if robust == "none" or beta <= 0:
        return mean
    if robust == "tv":
        clipped = np.minimum(values, float(np.min(values)) + beta)
        return float(np.dot(probs, clipped))
    if robust == "kl":
        safe_beta = max(beta, 1e-8)
        return float(-safe_beta * np.log(np.sum(probs * np.exp(-values / safe_beta)) + 1e-12))
    if robust == "chi2":
        centered = values - mean
        variance = float(np.dot(probs, centered * centered))
        return float(mean - np.sqrt(max(beta, 0.0) * max(variance, 0.0)))
    raise ValueError(f"Unsupported robust type: {robust}")


def tabular_bonus(robust, beta, count, horizon_left, scale=1.0):
    n = max(count, 1)
    base = scale * sqrt(log(1024.0) / (2.0 * n))
    if count == 0:
        base += horizon_left
    if robust == "tv":
        base += beta * sqrt(log(1024.0) / (2.0 * n))
    elif robust == "kl":
        base += 2.0 * beta * exp(horizon_left / max(beta, 1e-8)) * sqrt(log(1024.0) / (2.0 * n))
    elif robust == "chi2":
        base += 2.0 * sqrt(max(beta, 0.0)) * horizon_left * sqrt(log(1024.0) / (2.0 * n))
    else:
        base += horizon_left / sqrt(n)
    return base


def lower_tail_mean(values, tail_frac=0.25):
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return 0.0
    k = max(1, int(np.ceil(tail_frac * arr.size)))
    return float(np.mean(arr[:k]))


def _is_zero_sum_benchmark(benchmark):
    return benchmark.n_players == 2 and np.allclose(
        benchmark.rewards[..., 0] + benchmark.rewards[..., 1],
        0.0,
        atol=1e-8,
    )


def _stack_zero_sum_values(q_values):
    q_values = np.asarray(q_values, dtype=float)
    return np.stack([q_values, -q_values], axis=0)


def _project_zero_sum_q(q_state):
    q_state = np.asarray(q_state, dtype=float)
    if q_state.shape[0] != 2:
        return np.array(q_state, copy=True)
    center = 0.5 * (q_state[0] - q_state[1])
    return np.stack([center, -center], axis=0)


def _zero_sum_played_value(q_values, dist):
    return float(np.dot(dist, np.asarray(q_values, dtype=float)))


def _zero_sum_stage_gap(q_values, dist, action_sizes):
    q_state = _stack_zero_sum_values(q_values)
    played = _zero_sum_played_value(q_values, dist)
    row_best = float(np.max(_best_response_values(q_state, dist, action_sizes, 0)))
    col_best = float(np.max(_best_response_values(q_state, dist, action_sizes, 1)))
    return max(row_best - played, col_best + played)


def _best_response_values(q_state, dist, action_sizes, player):
    marginals = marginals_from_joint(dist, action_sizes)
    opp = 1 - player
    opp_marg = marginals[opp]
    acts = joint_actions(action_sizes)
    out = np.zeros(action_sizes[player], dtype=float)
    for act in range(action_sizes[player]):
        total = 0.0
        for opp_action, opp_prob in enumerate(opp_marg):
            joint = [0, 0]
            joint[player] = act
            joint[opp] = opp_action
            alt_idx = acts.index(tuple(joint))
            total += opp_prob * q_state[player, alt_idx]
        out[act] = total
    return out


def _player_entropy_bonus(dist, action_sizes, player, tau):
    if tau <= 0:
        return 0.0
    marginals = marginals_from_joint(dist, action_sizes)
    return float(tau * entropy(marginals[player]))


def _played_stage_value(q_state, dist, action_sizes, player, tau):
    return float(np.dot(dist, q_state[player])) + _player_entropy_bonus(dist, action_sizes, player, tau)


def _mixed_best_response_value(q_state, dist, action_sizes, player, tau):
    return soft_value(_best_response_values(q_state, dist, action_sizes, player), tau)


class TabularLearner:
    def __init__(self, benchmark, algo, seed=0):
        self.benchmark = benchmark
        self.algo = algo
        self.rng = set_seed(seed)
        h, s, a = benchmark.horizon, benchmark.state_count, benchmark.joint_action_count
        n = benchmark.n_players
        self.counts = np.zeros((h, s, a), dtype=int)
        self.reward_sums = np.zeros((h, s, a, n), dtype=float)
        self.transition_counts = np.zeros((h, s, a, s), dtype=float)
        self.policy = np.ones((h, s, a), dtype=float) / a
        self.last_q = np.zeros((n, h, s, a), dtype=float)
        self.last_stage_residual = 0.0

    def _plan(self):
        h = self.benchmark.horizon
        s_count = self.benchmark.state_count
        a_count = self.benchmark.joint_action_count
        n = self.benchmark.n_players
        if _is_zero_sum_benchmark(self.benchmark):
            v = np.zeros((h + 1, s_count), dtype=float)
            q_all = np.zeros((n, h, s_count, a_count), dtype=float)
            policy = np.zeros((h, s_count, a_count), dtype=float)
            max_stage_residual = 0.0
            for stage in range(h - 1, -1, -1):
                horizon_left = h - stage
                for state in range(s_count):
                    q_values = np.zeros(a_count, dtype=float)
                    for a_idx in range(a_count):
                        count = self.counts[stage, state, a_idx]
                        denom = max(count, 1)
                        reward_hat = self.reward_sums[stage, state, a_idx, 0] / denom
                        p_hat = (
                            self.transition_counts[stage, state, a_idx] / denom
                            if count > 0
                            else np.ones(s_count, dtype=float) / s_count
                        )
                        cont = robust_value(self.algo.robust, self.algo.beta, p_hat, v[stage + 1])
                        bonus = tabular_bonus(self.algo.robust, self.algo.beta, count, horizon_left, scale=self.algo.bonus_scale)
                        q_values[a_idx] = min(horizon_left, reward_hat + cont + bonus)
                    q_state = _stack_zero_sum_values(q_values)
                    dist = solve_stage_cce(
                        q_state,
                        self.benchmark.action_sizes,
                        tau=self.algo.tau,
                        violation_scale=self.algo.stage_violation_scale,
                        refine_steps=self.algo.stage_refine_steps,
                    )
                    policy[stage, state] = dist
                    v[stage, state] = _zero_sum_played_value(q_values, dist)
                    max_stage_residual = max(
                        max_stage_residual,
                        _zero_sum_stage_gap(q_values, dist, self.benchmark.action_sizes),
                    )
                    q_all[:, stage, state] = q_state
            self.policy = policy
            self.last_q = q_all
            self.last_stage_residual = float(max_stage_residual)
            return
        v = np.zeros((n, h + 1, s_count), dtype=float)
        w = np.zeros((n, h + 1, s_count), dtype=float)
        q_all = np.zeros((n, h, s_count, a_count), dtype=float)
        policy = np.zeros((h, s_count, a_count), dtype=float)
        max_stage_residual = 0.0
        for stage in range(h - 1, -1, -1):
            horizon_left = h - stage
            for state in range(s_count):
                q_state = np.zeros((n, a_count), dtype=float)
                for a_idx in range(a_count):
                    count = self.counts[stage, state, a_idx]
                    denom = max(count, 1)
                    r_hat = self.reward_sums[stage, state, a_idx] / denom
                    p_hat = self.transition_counts[stage, state, a_idx] / denom if count > 0 else np.ones(s_count) / s_count
                    for player in range(n):
                        cont = robust_value(self.algo.robust, self.algo.beta, p_hat, w[player, stage + 1])
                        bonus = tabular_bonus(self.algo.robust, self.algo.beta, count, horizon_left, scale=self.algo.bonus_scale)
                        q_state[player, a_idx] = min(horizon_left, r_hat[player] + cont + bonus)
                dist = solve_stage_cce(
                    q_state,
                    self.benchmark.action_sizes,
                    tau=self.algo.tau,
                    violation_scale=self.algo.stage_violation_scale,
                    refine_steps=self.algo.stage_refine_steps,
                )
                policy[stage, state] = dist
                for player in range(n):
                    played = _played_stage_value(q_state, dist, self.benchmark.action_sizes, player, self.algo.tau)
                    best = _mixed_best_response_value(q_state, dist, self.benchmark.action_sizes, player, self.algo.tau)
                    v[player, stage, state] = played
                    w[player, stage, state] = best
                    max_stage_residual = max(max_stage_residual, best - played)
                q_all[:, stage, state] = q_state
        self.policy = policy
        self.last_q = q_all
        self.last_stage_residual = float(max_stage_residual)

    def rollout(self, alpha=0.0, collect=False):
        state = self.benchmark.initial_state
        total = np.zeros(self.benchmark.n_players, dtype=float)
        traj = []
        fail_hits = 0
        for stage in range(self.benchmark.horizon):
            dist = clip_simplex(self.policy[stage, state])
            a_idx = int(self.rng.choice(self.benchmark.joint_action_count, p=dist))
            reward = self.benchmark.rewards[stage, state, a_idx]
            next_state = self.benchmark.sample_next_state(self.rng, stage, state, a_idx, alpha=alpha)
            total += reward
            fail_hits += int(next_state == self.benchmark.failure_state)
            if collect:
                traj.append((stage, state, a_idx, reward.copy(), next_state))
            state = next_state
        if collect:
            return traj
        value = float(total[0]) if _is_zero_sum_benchmark(self.benchmark) else float(total.mean())
        return value, fail_hits / self.benchmark.horizon

    def evaluate(self, alpha=0.0, return_details=False):
        exact_return, gap, one_shot_gap, fail_rate, details = evaluate_policy_exact(
            self.benchmark,
            self.policy,
            robust=self.algo.robust,
            beta=self.algo.beta,
            tau=self.algo.tau,
            alpha=alpha,
            return_details=True,
        )
        if return_details:
            return float(exact_return), float(gap), float(one_shot_gap), float(fail_rate), details
        return float(exact_return), float(gap), float(one_shot_gap), float(fail_rate)

    def compute_cce_gap(self, alpha=0.0):
        _, gap, _, _ = evaluate_policy_exact(
            self.benchmark,
            self.policy,
            robust=self.algo.robust,
            beta=self.algo.beta,
            tau=self.algo.tau,
            alpha=alpha,
        )
        return gap

    def compute_exploitability(self):
        _, gap, _, _ = evaluate_policy_exact(
            self.benchmark,
            self.policy,
            robust=self.algo.robust,
            beta=self.algo.beta,
            tau=self.algo.tau,
            alpha=0.0,
        )
        return gap

    def train(self, episodes, eval_every, alpha_grid):
        metrics = {
            "episodes": [],
            "returns": {str(alpha): [] for alpha in alpha_grid},
            "cce_gap": {str(alpha): [] for alpha in alpha_grid},
            "nash_gap": {str(alpha): [] for alpha in alpha_grid},
            "one_shot_gap": {str(alpha): [] for alpha in alpha_grid},
            "failure_rate": {str(alpha): [] for alpha in alpha_grid},
            "deployment_returns": {str(alpha): [] for alpha in alpha_grid},
            "deployment_nash_gap": {str(alpha): [] for alpha in alpha_grid},
            "deployment_failure_rate": {str(alpha): [] for alpha in alpha_grid},
            "regret": [],
            "approx_exploitability": [],
            "stage_residual": [],
            "solver_residual": [],
            "worst_case_return": [],
            "cvar_return": [],
            "deployment_worst_case_return": [],
            "deployment_cvar_return": [],
            "occupancy_entropy": [],
            "good_state_mass": [],
            "feature_gram_min_eig": [],
        }
        for ep in range(1, episodes + 1):
            if ep == 1 or (ep - 1) % self.algo.update_every == 0:
                self._plan()
            traj = self.rollout(alpha=0.0, collect=True)
            for stage, state, a_idx, rewards, next_state in traj:
                self.counts[stage, state, a_idx] += 1
                self.reward_sums[stage, state, a_idx] += rewards
                self.transition_counts[stage, state, a_idx, next_state] += 1
            if ep % eval_every == 0 or ep == episodes:
                metrics["episodes"].append(ep)
                grid_returns = []
                deployment_grid_returns = []
                nominal_details = None
                for alpha in alpha_grid:
                    ret, gap, one_shot_gap, fail_rate, details = self.evaluate(alpha, return_details=True)
                    dep_ret, dep_gap, _, dep_fail = evaluate_policy_exact(
                        self.benchmark,
                        self.policy,
                        robust="none",
                        beta=0.0,
                        tau=self.algo.tau,
                        alpha=alpha,
                    )
                    metrics["returns"][str(alpha)].append(ret)
                    metrics["cce_gap"][str(alpha)].append(gap)
                    metrics["nash_gap"][str(alpha)].append(gap)
                    metrics["one_shot_gap"][str(alpha)].append(one_shot_gap)
                    metrics["failure_rate"][str(alpha)].append(fail_rate)
                    metrics["deployment_returns"][str(alpha)].append(dep_ret)
                    metrics["deployment_nash_gap"][str(alpha)].append(dep_gap)
                    metrics["deployment_failure_rate"][str(alpha)].append(dep_fail)
                    grid_returns.append(ret)
                    deployment_grid_returns.append(dep_ret)
                    if alpha == 0.0 or nominal_details is None:
                        nominal_details = details
                exploitability = self.compute_exploitability()
                metrics["regret"].append(exploitability)
                metrics["approx_exploitability"].append(exploitability)
                metrics["stage_residual"].append(self.last_stage_residual)
                metrics["solver_residual"].append(self.last_stage_residual)
                metrics["worst_case_return"].append(float(min(grid_returns)))
                metrics["cvar_return"].append(lower_tail_mean(grid_returns))
                metrics["deployment_worst_case_return"].append(float(min(deployment_grid_returns)))
                metrics["deployment_cvar_return"].append(lower_tail_mean(deployment_grid_returns))
                metrics["occupancy_entropy"].append(float(nominal_details["occupancy_entropy"]))
                metrics["good_state_mass"].append(float(nominal_details["good_state_mass"]))
                metrics["feature_gram_min_eig"].append(0.0)
        return metrics


class LinearLearner(TabularLearner):
    def __init__(self, benchmark, algo, seed=0):
        super().__init__(benchmark, algo, seed=seed)
        self.feature_dim = benchmark.features.shape[-1]
        self.lam = algo.lam
        self.plan_round = 0
        self.phi_history = [[] for _ in range(benchmark.horizon)]
        self.reward_history = [[] for _ in range(benchmark.horizon)]
        self.next_state_history = [[] for _ in range(benchmark.horizon)]

    def _ridge_predictor(self, stage, targets):
        dim = self.feature_dim
        gram = self.lam * np.eye(dim)
        if not self.phi_history[stage]:
            return gram, np.zeros(dim, dtype=float)
        x = np.asarray(self.phi_history[stage], dtype=float)
        y = np.asarray(targets, dtype=float)
        gram = gram + x.T @ x
        weights = np.linalg.solve(gram, x.T @ y)
        return gram, weights

    def _plan(self):
        self.plan_round += 1
        h = self.benchmark.horizon
        s_count = self.benchmark.state_count
        a_count = self.benchmark.joint_action_count
        n = self.benchmark.n_players
        eta_k = self.algo.eta_scale / sqrt(max(self.plan_round, 1))
        xi_k = self.algo.xi_scale / sqrt(max(self.plan_round, 1))
        if _is_zero_sum_benchmark(self.benchmark):
            v = np.zeros((h + 1, s_count), dtype=float)
            q_all = np.zeros((n, h, s_count, a_count), dtype=float)
            policy = np.zeros((h, s_count, a_count), dtype=float)
            max_stage_residual = 0.0
            for stage in range(h - 1, -1, -1):
                reward_hist = np.asarray(self.reward_history[stage], dtype=float) if self.reward_history[stage] else np.zeros((0, n))
                gram, reward_theta = self._ridge_predictor(stage, reward_hist[:, 0] if len(reward_hist) else [])
                if self.next_state_history[stage]:
                    if self.algo.robust == "tv" and self.algo.beta > 0:
                        targets = [
                            min(v[stage + 1, s_next], float(np.min(v[stage + 1])) + self.algo.beta)
                            for s_next in self.next_state_history[stage]
                        ]
                    elif self.algo.robust == "kl" and self.algo.beta > 0:
                        targets = [
                            np.exp(-v[stage + 1, s_next] / max(self.algo.beta, 1e-8))
                            for s_next in self.next_state_history[stage]
                        ]
                    elif self.algo.robust == "chi2" and self.algo.beta > 0:
                        targets = [v[stage + 1, s_next] for s_next in self.next_state_history[stage]]
                    else:
                        targets = [v[stage + 1, s_next] for s_next in self.next_state_history[stage]]
                    _, trans_theta = self._ridge_predictor(stage, targets)
                    if self.algo.robust == "chi2" and self.algo.beta > 0:
                        sq_targets = [v[stage + 1, s_next] ** 2 for s_next in self.next_state_history[stage]]
                        _, trans_sq_theta = self._ridge_predictor(stage, sq_targets)
                    else:
                        trans_sq_theta = np.zeros(self.feature_dim, dtype=float)
                else:
                    trans_theta = np.zeros(self.feature_dim, dtype=float)
                    trans_sq_theta = np.zeros(self.feature_dim, dtype=float)
                for state in range(s_count):
                    q_values = np.zeros(a_count, dtype=float)
                    for a_idx in range(a_count):
                        phi = self.benchmark.features[state, a_idx]
                        horizon_left = h - stage
                        reward_est = float(phi @ reward_theta)
                        if self.algo.robust == "kl" and self.algo.beta > 0:
                            pred = max(float(phi @ trans_theta), np.exp(-horizon_left / max(self.algo.beta, 1e-8)))
                            trans_est = float(-self.algo.beta * np.log(pred))
                            scale = 1.5 + 2.0 * self.algo.beta * np.exp(horizon_left / max(self.algo.beta, 1e-8))
                        elif self.algo.robust == "chi2" and self.algo.beta > 0:
                            pred_mean = float(phi @ trans_theta)
                            pred_sq = float(phi @ trans_sq_theta)
                            pred_var = max(pred_sq - pred_mean**2, 0.0)
                            trans_est = float(pred_mean - np.sqrt(max(self.algo.beta, 0.0) * pred_var))
                            scale = 1.5 + 2.0 * np.sqrt(max(self.algo.beta, 0.0)) * horizon_left
                        else:
                            trans_est = float(phi @ trans_theta)
                            scale = 1.5 + (self.algo.beta if self.algo.robust == "tv" else 0.0)
                        psi = float(np.sqrt(phi @ np.linalg.solve(gram, phi)))
                        q_values[a_idx] = min(horizon_left, reward_est + trans_est + scale * psi)
                    q_state = _stack_zero_sum_values(q_values)
                    q_solver = _project_zero_sum_q(discretize_stage_game(q_state, eta_k))
                    dist = solve_stage_cce(
                        q_solver,
                        self.benchmark.action_sizes,
                        tau=self.algo.tau,
                        approx_tol=xi_k,
                        violation_scale=self.algo.stage_violation_scale,
                        refine_steps=self.algo.stage_refine_steps,
                    )
                    policy[stage, state] = dist
                    v[stage, state] = _zero_sum_played_value(q_values, dist)
                    max_stage_residual = max(
                        max_stage_residual,
                        _zero_sum_stage_gap(q_values, dist, self.benchmark.action_sizes),
                    )
                    q_all[:, stage, state] = q_state
            self.policy = policy
            self.last_q = q_all
            self.last_stage_residual = float(max_stage_residual)
            return
        v = np.zeros((n, h + 1, s_count), dtype=float)
        w = np.zeros((n, h + 1, s_count), dtype=float)
        q_all = np.zeros((n, h, s_count, a_count), dtype=float)
        policy = np.zeros((h, s_count, a_count), dtype=float)
        max_stage_residual = 0.0
        for stage in range(h - 1, -1, -1):
            reward_hist = np.asarray(self.reward_history[stage], dtype=float) if self.reward_history[stage] else np.zeros((0, n))
            grams = []
            reward_models = []
            for player in range(n):
                gram, theta = self._ridge_predictor(stage, reward_hist[:, player] if len(reward_hist) else [])
                grams.append(gram)
                reward_models.append(theta)
            trans_models = []
            trans_second_models = []
            for player in range(n):
                if self.next_state_history[stage]:
                    if self.algo.robust == "tv" and self.algo.beta > 0:
                        targets = [
                            min(w[player, stage + 1, s_next], float(np.min(w[player, stage + 1])) + self.algo.beta)
                            for s_next in self.next_state_history[stage]
                        ]
                    elif self.algo.robust == "kl" and self.algo.beta > 0:
                        targets = [
                            np.exp(-w[player, stage + 1, s_next] / max(self.algo.beta, 1e-8))
                            for s_next in self.next_state_history[stage]
                        ]
                    elif self.algo.robust == "chi2" and self.algo.beta > 0:
                        targets = [w[player, stage + 1, s_next] for s_next in self.next_state_history[stage]]
                    else:
                        targets = [w[player, stage + 1, s_next] for s_next in self.next_state_history[stage]]
                    _, trans_theta = self._ridge_predictor(stage, targets)
                    if self.algo.robust == "chi2" and self.algo.beta > 0:
                        sq_targets = [w[player, stage + 1, s_next] ** 2 for s_next in self.next_state_history[stage]]
                        _, trans_sq_theta = self._ridge_predictor(stage, sq_targets)
                    else:
                        trans_sq_theta = np.zeros(self.feature_dim, dtype=float)
                else:
                    trans_theta = np.zeros(self.feature_dim, dtype=float)
                    trans_sq_theta = np.zeros(self.feature_dim, dtype=float)
                trans_models.append(trans_theta)
                trans_second_models.append(trans_sq_theta)
            for state in range(s_count):
                q_state = np.zeros((n, a_count), dtype=float)
                for a_idx in range(a_count):
                    phi = self.benchmark.features[state, a_idx]
                    for player in range(n):
                        horizon_left = h - stage
                        reward_est = float(phi @ reward_models[player])
                        if self.algo.robust == "kl" and self.algo.beta > 0:
                            pred = max(float(phi @ trans_models[player]), np.exp(-horizon_left / max(self.algo.beta, 1e-8)))
                            trans_est = float(-self.algo.beta * np.log(pred))
                            scale = 1.5 + 2.0 * self.algo.beta * np.exp(horizon_left / max(self.algo.beta, 1e-8))
                        elif self.algo.robust == "chi2" and self.algo.beta > 0:
                            pred_mean = float(phi @ trans_models[player])
                            pred_sq = float(phi @ trans_second_models[player])
                            pred_var = max(pred_sq - pred_mean**2, 0.0)
                            trans_est = float(pred_mean - np.sqrt(max(self.algo.beta, 0.0) * pred_var))
                            scale = 1.5 + 2.0 * np.sqrt(max(self.algo.beta, 0.0)) * horizon_left
                        else:
                            trans_est = float(phi @ trans_models[player])
                            scale = 1.5 + (self.algo.beta if self.algo.robust == "tv" else 0.0)
                        psi = float(np.sqrt(phi @ np.linalg.solve(grams[player], phi)))
                        q_state[player, a_idx] = min(horizon_left, reward_est + trans_est + scale * psi)
                q_solver = discretize_stage_game(q_state, eta_k)
                dist = solve_stage_cce(
                    q_solver,
                    self.benchmark.action_sizes,
                    tau=self.algo.tau,
                    approx_tol=xi_k,
                    violation_scale=self.algo.stage_violation_scale,
                    refine_steps=self.algo.stage_refine_steps,
                )
                policy[stage, state] = dist
                for player in range(n):
                    played = _played_stage_value(q_state, dist, self.benchmark.action_sizes, player, self.algo.tau)
                    best = _mixed_best_response_value(q_state, dist, self.benchmark.action_sizes, player, self.algo.tau)
                    v[player, stage, state] = played
                    w[player, stage, state] = best
                    max_stage_residual = max(max_stage_residual, best - played)
                q_all[:, stage, state] = q_state
        self.policy = policy
        self.last_q = q_all
        self.last_stage_residual = float(max_stage_residual)

    def feature_gram_min_eig(self):
        eigs = []
        for stage in range(self.benchmark.horizon):
            if not self.phi_history[stage]:
                eigs.append(0.0)
                continue
            features = np.asarray(self.phi_history[stage], dtype=float)
            gram = (features.T @ features) / max(features.shape[0], 1)
            eigs.append(float(np.min(np.linalg.eigvalsh(gram))))
        return float(min(eigs)) if eigs else 0.0

    def train(self, episodes, eval_every, alpha_grid):
        metrics = {
            "episodes": [],
            "returns": {str(alpha): [] for alpha in alpha_grid},
            "cce_gap": {str(alpha): [] for alpha in alpha_grid},
            "nash_gap": {str(alpha): [] for alpha in alpha_grid},
            "one_shot_gap": {str(alpha): [] for alpha in alpha_grid},
            "failure_rate": {str(alpha): [] for alpha in alpha_grid},
            "deployment_returns": {str(alpha): [] for alpha in alpha_grid},
            "deployment_nash_gap": {str(alpha): [] for alpha in alpha_grid},
            "deployment_failure_rate": {str(alpha): [] for alpha in alpha_grid},
            "regret": [],
            "approx_exploitability": [],
            "stage_residual": [],
            "solver_residual": [],
            "worst_case_return": [],
            "cvar_return": [],
            "deployment_worst_case_return": [],
            "deployment_cvar_return": [],
            "occupancy_entropy": [],
            "good_state_mass": [],
            "feature_gram_min_eig": [],
        }
        for ep in range(1, episodes + 1):
            if ep == 1 or (ep - 1) % self.algo.update_every == 0:
                self._plan()
            traj = self.rollout(alpha=0.0, collect=True)
            for stage, state, a_idx, rewards, next_state in traj:
                self.counts[stage, state, a_idx] += 1
                self.reward_sums[stage, state, a_idx] += rewards
                self.transition_counts[stage, state, a_idx, next_state] += 1
                self.phi_history[stage].append(self.benchmark.features[state, a_idx].copy())
                self.reward_history[stage].append(rewards.copy())
                self.next_state_history[stage].append(next_state)
            if ep % eval_every == 0 or ep == episodes:
                metrics["episodes"].append(ep)
                grid_returns = []
                deployment_grid_returns = []
                nominal_details = None
                for alpha in alpha_grid:
                    ret, gap, one_shot_gap, fail_rate, details = self.evaluate(alpha, return_details=True)
                    dep_ret, dep_gap, _, dep_fail = evaluate_policy_exact(
                        self.benchmark,
                        self.policy,
                        robust="none",
                        beta=0.0,
                        tau=self.algo.tau,
                        alpha=alpha,
                    )
                    metrics["returns"][str(alpha)].append(ret)
                    metrics["cce_gap"][str(alpha)].append(gap)
                    metrics["nash_gap"][str(alpha)].append(gap)
                    metrics["one_shot_gap"][str(alpha)].append(one_shot_gap)
                    metrics["failure_rate"][str(alpha)].append(fail_rate)
                    metrics["deployment_returns"][str(alpha)].append(dep_ret)
                    metrics["deployment_nash_gap"][str(alpha)].append(dep_gap)
                    metrics["deployment_failure_rate"][str(alpha)].append(dep_fail)
                    grid_returns.append(ret)
                    deployment_grid_returns.append(dep_ret)
                    if alpha == 0.0 or nominal_details is None:
                        nominal_details = details
                exploitability = self.compute_exploitability()
                metrics["regret"].append(exploitability)
                metrics["approx_exploitability"].append(exploitability)
                metrics["stage_residual"].append(self.last_stage_residual)
                metrics["solver_residual"].append(self.last_stage_residual)
                metrics["worst_case_return"].append(float(min(grid_returns)))
                metrics["cvar_return"].append(lower_tail_mean(grid_returns))
                metrics["deployment_worst_case_return"].append(float(min(deployment_grid_returns)))
                metrics["deployment_cvar_return"].append(lower_tail_mean(deployment_grid_returns))
                metrics["occupancy_entropy"].append(float(nominal_details["occupancy_entropy"]))
                metrics["good_state_mass"].append(float(nominal_details["good_state_mass"]))
                metrics["feature_gram_min_eig"].append(self.feature_gram_min_eig())
        return metrics


def make_learner(
    benchmark,
    algo_name,
    seed=0,
    beta_override=None,
    tau_override=None,
    lam_override=None,
    eta_scale_override=None,
    xi_scale_override=None,
    stage_violation_scale_override=None,
    stage_refine_steps_override=None,
    update_every_override=None,
):
    algo = make_algo_config(
        algo_name,
        beta_override=beta_override,
        tau_override=tau_override,
        lam_override=lam_override,
        eta_scale_override=eta_scale_override,
        xi_scale_override=xi_scale_override,
        stage_violation_scale_override=stage_violation_scale_override,
        stage_refine_steps_override=stage_refine_steps_override,
        update_every_override=update_every_override,
    )
    if algo.linear:
        return LinearLearner(benchmark, algo, seed=seed), algo
    return TabularLearner(benchmark, algo, seed=seed), algo


def evaluate_policy_exact(benchmark, policy, robust="none", beta=0.0, tau=0.0, alpha=0.0, return_details=False):
    n = benchmark.n_players
    h = benchmark.horizon
    s_count = benchmark.state_count
    a_count = benchmark.joint_action_count
    if _is_zero_sum_benchmark(benchmark):
        q_play = np.zeros((h, s_count, a_count), dtype=float)
        v_play = np.zeros((h + 1, s_count), dtype=float)
        occ = np.zeros((h, s_count), dtype=float)
        occ[0, benchmark.initial_state] = 1.0
        for stage in range(h - 1, -1, -1):
            for state in range(s_count):
                dist = clip_simplex(policy[stage, state])
                for a_idx in range(a_count):
                    probs = benchmark.next_transition(stage, state, a_idx, alpha=alpha)
                    q_play[stage, state, a_idx] = benchmark.rewards[stage, state, a_idx, 0] + robust_value(
                        robust,
                        beta,
                        probs,
                        v_play[stage + 1],
                    )
                v_play[stage, state] = _zero_sum_played_value(q_play[stage, state], dist)
        for stage in range(h - 1):
            for state in range(s_count):
                if occ[stage, state] <= 0:
                    continue
                dist = clip_simplex(policy[stage, state])
                for a_idx in range(a_count):
                    trans = benchmark.next_transition(stage, state, a_idx, alpha=alpha)
                    occ[stage + 1] += occ[stage, state] * dist[a_idx] * trans
        failure_rate = float(np.sum(occ[:, benchmark.failure_state]) / h)
        joint_occupancy = []
        for stage in range(h):
            for state in range(s_count):
                if occ[stage, state] <= 0:
                    continue
                dist = clip_simplex(policy[stage, state])
                for action_prob in dist:
                    mass = occ[stage, state] * action_prob
                    if mass > 0:
                        joint_occupancy.append(float(mass))
        occ_arr = np.asarray(joint_occupancy, dtype=float)
        occupancy_entropy = float(-np.sum(occ_arr * np.log(np.maximum(occ_arr, 1e-12)))) if occ_arr.size else 0.0
        good_state_mask = np.ones(s_count, dtype=bool)
        good_state_mask[benchmark.failure_state] = False
        good_state_mass = float(np.sum(occ[:, good_state_mask]) / h)
        init_dist = clip_simplex(policy[0, benchmark.initial_state])
        init_played = _zero_sum_played_value(q_play[0, benchmark.initial_state], init_dist)
        max_gap = float(_zero_sum_stage_gap(q_play[0, benchmark.initial_state], init_dist, benchmark.action_sizes))
        one_shot_gap = 0.0
        for stage in range(h):
            for state in range(s_count):
                if occ[stage, state] <= 0:
                    continue
                dist = clip_simplex(policy[stage, state])
                one_shot_gap += occ[stage, state] * _zero_sum_stage_gap(
                    q_play[stage, state],
                    dist,
                    benchmark.action_sizes,
                )
        result = (
            float(init_played),
            float(max_gap),
            float(one_shot_gap),
            failure_rate,
        )
        if not return_details:
            return result
        return result + (
            {
                "occupancy_entropy": occupancy_entropy,
                "good_state_mass": good_state_mass,
            },
        )
    q_play = np.zeros((n, h, s_count, a_count), dtype=float)
    q_br = np.zeros((n, h, s_count, a_count), dtype=float)
    v_play = np.zeros((n, h + 1, s_count), dtype=float)
    w_br = np.zeros((n, h + 1, s_count), dtype=float)
    occ = np.zeros((h, s_count), dtype=float)
    occ[0, benchmark.initial_state] = 1.0
    for stage in range(h - 1, -1, -1):
        for state in range(s_count):
            dist = clip_simplex(policy[stage, state])
            for a_idx in range(a_count):
                probs = benchmark.next_transition(stage, state, a_idx, alpha=alpha)
                for player in range(n):
                    q_play[player, stage, state, a_idx] = benchmark.rewards[stage, state, a_idx, player] + robust_value(
                        robust,
                        beta,
                        probs,
                        v_play[player, stage + 1],
                    )
                    q_br[player, stage, state, a_idx] = benchmark.rewards[stage, state, a_idx, player] + robust_value(
                        robust,
                        beta,
                        probs,
                        w_br[player, stage + 1],
                    )
            for player in range(n):
                v_play[player, stage, state] = _played_stage_value(
                    q_play[:, stage, state],
                    dist,
                    benchmark.action_sizes,
                    player,
                    tau,
                )
                w_br[player, stage, state] = _mixed_best_response_value(
                    q_br[:, stage, state],
                    dist,
                    benchmark.action_sizes,
                    player,
                    tau,
                )
        if stage < h - 1:
            continue
    for stage in range(h - 1):
        for state in range(s_count):
            if occ[stage, state] <= 0:
                continue
            for a_idx in range(a_count):
                trans = benchmark.next_transition(stage, state, a_idx, alpha=alpha)
                occ[stage + 1] += occ[stage, state] * clip_simplex(policy[stage, state])[a_idx] * trans
    failure_rate = float(np.sum(occ[:, benchmark.failure_state]) / h)
    joint_occupancy = []
    for stage in range(h):
        for state in range(s_count):
            if occ[stage, state] <= 0:
                continue
            dist = clip_simplex(policy[stage, state])
            for action_prob in dist:
                mass = occ[stage, state] * action_prob
                if mass > 0:
                    joint_occupancy.append(float(mass))
    occ_arr = np.asarray(joint_occupancy, dtype=float)
    occupancy_entropy = float(-np.sum(occ_arr * np.log(np.maximum(occ_arr, 1e-12)))) if occ_arr.size else 0.0
    good_state_mask = np.ones(s_count, dtype=bool)
    good_state_mask[benchmark.failure_state] = False
    good_state_mass = float(np.sum(occ[:, good_state_mask]) / h)
    max_gap = 0.0
    max_one_shot_gap = 0.0
    for player in range(n):
        dyn_gap = w_br[player, 0, benchmark.initial_state] - v_play[player, 0, benchmark.initial_state]
        max_gap = max(max_gap, dyn_gap)
        one_shot_gap = 0.0
        for stage in range(h):
            for state in range(s_count):
                if occ[stage, state] <= 0:
                    continue
                dist = clip_simplex(policy[stage, state])
                played = _played_stage_value(
                    q_play[:, stage, state],
                    dist,
                    benchmark.action_sizes,
                    player,
                    tau,
                )
                br_val = _mixed_best_response_value(
                    q_play[:, stage, state],
                    dist,
                    benchmark.action_sizes,
                    player,
                    tau,
                )
                one_shot_gap += occ[stage, state] * max(0.0, br_val - played)
        max_one_shot_gap = max(max_one_shot_gap, one_shot_gap)
    result = (
        float(v_play[:, 0, benchmark.initial_state].mean()),
        float(max_gap),
        float(max_one_shot_gap),
        failure_rate,
    )
    if not return_details:
        return result
    return result + (
        {
            "occupancy_entropy": occupancy_entropy,
            "good_state_mass": good_state_mass,
        },
    )
