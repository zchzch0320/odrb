from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import clip_simplex, joint_actions


@dataclass
class BenchmarkSpec:
    name: str
    horizon: int
    state_count: int
    action_sizes: tuple
    initial_state: int
    failure_state: int
    terminal_state: int | None
    rewards: np.ndarray
    transitions_nominal: np.ndarray
    alpha_grid: list[float]
    features: np.ndarray | None = None
    anchor_kernels: np.ndarray | None = None
    overlap_m: float | None = None
    paper_name: str | None = None

    @property
    def n_players(self):
        return len(self.action_sizes)

    @property
    def joint_actions(self):
        return joint_actions(self.action_sizes)

    @property
    def joint_action_count(self):
        return len(self.joint_actions)

    def next_transition(self, h, state, action_idx, alpha=0.0):
        probs = np.array(self.transitions_nominal[h, state, action_idx], copy=True)
        if alpha > 0 and self.name == "e1":
            if state in (2, 3):
                action = self.joint_actions[action_idx]
                if action == (1, 1):
                    fail_prob = min(1.0, 0.10 + alpha)
                else:
                    fail_prob = min(1.0, 0.04 + 0.5 * alpha)
                safe_target = 3 if state == 2 else 5
                probs[:] = 0.0
                probs[self.failure_state] = fail_prob
                probs[safe_target] = 1.0 - fail_prob
        elif alpha > 0 and self.name == "e2":
            if state not in (5, 6):
                a0, a1 = self.joint_actions[action_idx]
                congestion = 2 if (a0 == 0 and a1 == 0) else 1 if (a0 == 0 or a1 == 0) else 0
                fail_prob = min(1.0, 0.03 + 0.06 * congestion + alpha * (1 if congestion >= 1 else 0))
                next_state = state + 1 if state + 1 < self.state_count - 1 else self.terminal_state
                probs[:] = 0.0
                probs[self.failure_state] = fail_prob
                probs[next_state] = 1.0 - fail_prob
        elif alpha > 0 and self.name == "e3":
            if self.features[state, action_idx, 0] + self.features[state, action_idx, 1] >= 0.35 and state != self.failure_state:
                probs = (1.0 - alpha) * probs
                probs[self.failure_state] += alpha
                probs = probs / probs.sum()
        elif alpha > 0 and self.name == "e4":
            if state == 0:
                action = self.joint_actions[action_idx]
                if action[0] == 1:
                    base_bad = float(self.overlap_m or 0.0)
                    shifted_bad = min(1.0, base_bad + alpha * (1.0 - base_bad))
                    probs[:] = 0.0
                    probs[1] = 1.0 - shifted_bad
                    probs[2] = shifted_bad
        return probs

    def sample_next_state(self, rng, h, state, action_idx, alpha=0.0):
        probs = clip_simplex(self.next_transition(h, state, action_idx, alpha=alpha))
        return int(rng.choice(self.state_count, p=probs))


def _zero_sumify(rewards: np.ndarray) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=float)
    out = np.array(rewards, copy=True)
    out[..., 1] = -out[..., 0]
    return out


def build_e1():
    horizon = 4
    state_count = 6
    action_sizes = (2, 2)
    joint = joint_actions(action_sizes)
    rewards = np.zeros((horizon, state_count, len(joint), 2), dtype=float)
    transitions = np.zeros((horizon, state_count, len(joint), state_count), dtype=float)
    for h in range(horizon):
        for a_idx, (a0, a1) in enumerate(joint):
            if a0 == 1 and a1 == 1:
                rewards[h, 0, a_idx, :] = 0.75
                transitions[h, 0, a_idx, 2] = 1.0
            else:
                rewards[h, 0, a_idx, :] = 0.45
                transitions[h, 0, a_idx, 1] = 1.0
            rewards[h, 1, a_idx, :] = 0.35
            transitions[h, 1, a_idx, 5] = 1.0
            if (a0, a1) == (1, 1):
                rewards[h, 2, a_idx, :] = 1.0
                rewards[h, 3, a_idx, :] = 1.0
                transitions[h, 2, a_idx, 3] = 0.90
                transitions[h, 2, a_idx, 4] = 0.10
                transitions[h, 3, a_idx, 5] = 0.90
                transitions[h, 3, a_idx, 4] = 0.10
            elif (a0, a1) == (0, 0):
                rewards[h, 2, a_idx, :] = 0.55
                rewards[h, 3, a_idx, :] = 0.55
                transitions[h, 2, a_idx, 5] = 0.96
                transitions[h, 2, a_idx, 4] = 0.04
                transitions[h, 3, a_idx, 5] = 0.96
                transitions[h, 3, a_idx, 4] = 0.04
            else:
                rewards[h, 2, a_idx, :] = 0.70
                rewards[h, 3, a_idx, :] = 0.70
                transitions[h, 2, a_idx, 5] = 0.96
                transitions[h, 2, a_idx, 4] = 0.04
                transitions[h, 3, a_idx, 5] = 0.96
                transitions[h, 3, a_idx, 4] = 0.04
            transitions[h, 4, a_idx, 4] = 1.0
            transitions[h, 5, a_idx, 5] = 1.0
    rewards = _zero_sumify(rewards)
    return BenchmarkSpec(
        name="e1",
        horizon=horizon,
        state_count=state_count,
        action_sizes=action_sizes,
        initial_state=0,
        failure_state=4,
        terminal_state=5,
        rewards=rewards,
        transitions_nominal=transitions,
        alpha_grid=[round(0.05 * i, 2) for i in range(9)],
        paper_name="tabular_attack_defense",
    )


def build_e2():
    horizon = 5
    state_count = 7
    action_sizes = (3, 3)
    joint = joint_actions(action_sizes)
    rewards = np.zeros((horizon, state_count, len(joint), 2), dtype=float)
    transitions = np.zeros((horizon, state_count, len(joint), state_count), dtype=float)
    base_u = {0: 1.0, 1: 0.8, 2: 0.55}
    for h in range(horizon):
        for s in range(5):
            for a_idx, (a0, a1) in enumerate(joint):
                congestion = 2 if (a0 == 0 and a1 == 0) else 1 if (a0 == 0 or a1 == 0) else 0
                fail_prob = min(1.0, 0.03 + 0.06 * congestion)
                next_state = s + 1 if s < 4 else 6
                transitions[h, s, a_idx, 5] = fail_prob
                transitions[h, s, a_idx, next_state] = 1.0 - fail_prob
                penalty_fast = 0.20 if (a0 == 0 and a1 == 0) else 0.0
                penalty_mid = 0.10 if (a0 == 1 and a1 == 1) else 0.0
                rewards[h, s, a_idx, 0] = max(0.0, base_u[a0] - penalty_fast - penalty_mid)
                rewards[h, s, a_idx, 1] = max(0.0, base_u[a1] - penalty_fast - penalty_mid)
        for a_idx in range(len(joint)):
            transitions[h, 5, a_idx, 5] = 1.0
            transitions[h, 6, a_idx, 6] = 1.0
    rewards = _zero_sumify(rewards)
    return BenchmarkSpec(
        name="e2",
        horizon=horizon,
        state_count=state_count,
        action_sizes=action_sizes,
        initial_state=0,
        failure_state=5,
        terminal_state=6,
        rewards=rewards,
        transitions_nominal=transitions,
        alpha_grid=[round(0.05 * i, 2) for i in range(9)],
        paper_name="tabular_queue_guard",
    )


def build_e3(seed=0, dim=16):
    rng = np.random.default_rng(seed)
    horizon = 3
    state_count = 6
    action_sizes = (2, 2)
    joint = joint_actions(action_sizes)
    joint_count = len(joint)
    features = np.zeros((state_count, joint_count, dim), dtype=float)
    anchor = np.zeros((horizon, dim, state_count), dtype=float)
    rewards = np.zeros((horizon, state_count, joint_count, 2), dtype=float)
    transitions = np.zeros((horizon, state_count, joint_count, state_count), dtype=float)

    for h in range(horizon):
        for j in range(dim):
            if j < 2:
                anchor[h, j, 5] = 0.40
                anchor[h, j, 1:5] = 0.15
            elif j < 4:
                anchor[h, j, 5] = 0.10
                anchor[h, j, 1:5] = 0.225
            else:
                raw = rng.dirichlet(np.ones(state_count))
                raw[5] = min(raw[5], 0.05)
                raw = raw / raw.sum()
                anchor[h, j] = raw
        theta = rng.normal(size=(2, dim))
        theta = theta / np.maximum(np.linalg.norm(theta, axis=1, keepdims=True), 1e-8) * 1.5
        for s in range(state_count):
            for a_idx in range(joint_count):
                raw = np.zeros(dim)
                active = rng.choice(dim, size=min(3, dim), replace=False)
                raw[active] = rng.normal(loc=0.0, scale=1.0, size=len(active))
                weights = np.exp(raw - raw.max())
                weights = weights / weights.sum()
                features[s, a_idx] = weights
                phi = features[s, a_idx]
                r = theta @ phi
                bonus = 0.20 if phi[0] + phi[1] >= 0.35 else 0.0
                rewards[h, s, a_idx, 0] = float(np.clip(r[0] + bonus, 0.0, 1.0))
                rewards[h, s, a_idx, 1] = float(np.clip(r[1] + 0.5 * bonus, 0.0, 1.0))
                transitions[h, s, a_idx] = phi @ anchor[h]
        for a_idx in range(joint_count):
            transitions[h, 5, a_idx, :] = 0.0
            transitions[h, 5, a_idx, 5] = 1.0
            rewards[h, 5, a_idx, :] = 0.0
    rewards = _zero_sumify(rewards)
    return BenchmarkSpec(
        name="e3",
        horizon=horizon,
        state_count=state_count,
        action_sizes=action_sizes,
        initial_state=0,
        failure_state=5,
        terminal_state=None,
        rewards=rewards,
        transitions_nominal=transitions,
        alpha_grid=[round(0.02 * i, 2) for i in range(11)],
        features=features,
        anchor_kernels=anchor,
        paper_name="linear_mixture",
    )


def build_e4(overlap_m=0.10):
    m = float(overlap_m)
    horizon = 4
    state_count = 6
    action_sizes = (2, 2)
    joint = joint_actions(action_sizes)
    rewards = np.zeros((horizon, state_count, len(joint), 2), dtype=float)
    transitions = np.zeros((horizon, state_count, len(joint), state_count), dtype=float)

    for h in range(horizon):
        for a_idx, action in enumerate(joint):
            if action == (0, 0):
                rewards[h, 0, a_idx, :] = 0.16
                transitions[h, 0, a_idx, 3] = 1.0
            elif action == (0, 1):
                rewards[h, 0, a_idx, :] = 0.13
                transitions[h, 0, a_idx, 3] = 1.0
            elif action == (1, 0):
                rewards[h, 0, a_idx, :] = 0.12
                transitions[h, 0, a_idx, 1] = 1.0 - m
                transitions[h, 0, a_idx, 2] = m
            else:
                rewards[h, 0, a_idx, :] = 0.09
                transitions[h, 0, a_idx, 1] = 1.0 - m
                transitions[h, 0, a_idx, 2] = m

            rewards[h, 1, a_idx, :] = 0.80
            transitions[h, 1, a_idx, 5] = 1.0

            rewards[h, 2, a_idx, :] = -0.35
            transitions[h, 2, a_idx, 4] = 1.0

            rewards[h, 3, a_idx, :] = 0.44
            transitions[h, 3, a_idx, 5] = 1.0

            transitions[h, 4, a_idx, 4] = 1.0
            transitions[h, 5, a_idx, 5] = 1.0

    rewards = _zero_sumify(rewards)
    return BenchmarkSpec(
        name="e4",
        horizon=horizon,
        state_count=state_count,
        action_sizes=action_sizes,
        initial_state=0,
        failure_state=4,
        terminal_state=5,
        rewards=rewards,
        transitions_nominal=transitions,
        alpha_grid=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        overlap_m=m,
        paper_name="support_shift",
    )


def make_benchmark(name, seed=0, dim=16, overlap_m=0.10):
    if name in {"e1", "tabular_attack_defense"}:
        return build_e1()
    if name in {"e2", "tabular_queue_guard"}:
        return build_e2()
    if name in {"e3", "linear_mixture"}:
        return build_e3(seed=seed, dim=dim)
    if name in {"e4", "support_shift"}:
        return build_e4(overlap_m=overlap_m)
    raise ValueError(f"Unknown benchmark: {name}")
