from __future__ import annotations

import argparse
from pathlib import Path

from gfa.benchmarks import make_benchmark
from gfa.learners import make_learner
from gfa.utils import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run theorem-aligned robust zero-sum Markov-game experiments.")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=[
            "tabular_attack_defense",
            "tabular_queue_guard",
            "linear_mixture",
            "support_shift",
            "e1",
            "e2",
            "e3",
            "e4",
        ],
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=[
            "nonrobust_cce",
            "policyreg_only",
            "robust_tv",
            "robust_kl",
            "robust_chi2",
            "drmg_tv",
            "drmg_kl",
            "drmg_chi2",
            "nqovi_l",
            "policyreg_l",
            "dr_cce_lsi",
            "dr_cce_lsi_kl",
            "dr_cce_lsi_chi2",
            "drmg_tv_l",
            "drmg_kl_l",
            "drmg_chi2_l",
            "route_a_tabular",
            "sigma0_tabular",
            "no_exploiter_tabular",
            "route_a_linear",
            "sigma0_linear",
            "no_exploiter_linear",
        ],
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--eta-scale", type=float, default=None)
    parser.add_argument("--xi-scale", type=float, default=None)
    parser.add_argument("--stage-violation-scale", type=float, default=None)
    parser.add_argument("--stage-refine-steps", type=int, default=None)
    parser.add_argument("--update-every", type=int, default=None)
    parser.add_argument("--overlap-m", type=float, default=0.10)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = make_benchmark(args.benchmark, seed=args.seed, dim=args.dim, overlap_m=args.overlap_m)
    learner, algo = make_learner(
        benchmark,
        args.algorithm,
        seed=args.seed,
        beta_override=args.beta,
        tau_override=args.tau,
        lam_override=args.lam,
        eta_scale_override=args.eta_scale,
        xi_scale_override=args.xi_scale,
        stage_violation_scale_override=args.stage_violation_scale,
        stage_refine_steps_override=args.stage_refine_steps,
        update_every_override=args.update_every,
    )
    metrics = learner.train(
        episodes=args.episodes,
        eval_every=args.eval_every,
        alpha_grid=benchmark.alpha_grid,
    )
    stage0_policy = None
    stage0_player0_mass = None
    stage0_player1_mass = None
    if hasattr(learner, "policy"):
        stage0 = learner.policy[0, benchmark.initial_state]
        stage0_policy = [float(x) for x in stage0]
        stage0_player0_mass = [0.0 for _ in range(benchmark.action_sizes[0])]
        stage0_player1_mass = [0.0 for _ in range(benchmark.action_sizes[1])]
        for a_idx, joint_action in enumerate(benchmark.joint_actions):
            stage0_player0_mass[joint_action[0]] += stage0_policy[a_idx]
            stage0_player1_mass[joint_action[1]] += stage0_policy[a_idx]
    payload = {
        "benchmark": args.benchmark,
        "paper_benchmark": getattr(benchmark, "paper_name", args.benchmark),
        "algorithm": algo.name,
        "episodes": args.episodes,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "dim": args.dim,
        "beta": algo.beta,
        "tau": algo.tau,
        "lam": algo.lam,
        "eta_scale": algo.eta_scale,
        "xi_scale": algo.xi_scale,
        "stage_violation_scale": algo.stage_violation_scale,
        "stage_refine_steps": algo.stage_refine_steps,
        "tag": args.tag,
        "overlap_m": getattr(benchmark, "overlap_m", None),
        "alpha_grid": benchmark.alpha_grid,
        "joint_actions": [list(action) for action in benchmark.joint_actions],
        "stage0_joint_policy": stage0_policy,
        "stage0_player0_action_mass": stage0_player0_mass,
        "stage0_player1_action_mass": stage0_player1_mass,
        "metrics": metrics,
    }
    algo_dir = args.algorithm if not args.tag else f"{args.algorithm}__{args.tag}"
    out_dir = Path(args.output_dir) / args.benchmark / algo_dir
    ensure_dir(out_dir)
    out_file = out_dir / f"seed_{args.seed}.json"
    save_json(out_file, payload)
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
