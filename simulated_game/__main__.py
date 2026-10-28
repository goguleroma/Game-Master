import argparse
from pathlib import Path

from simulated_game.runner import run_game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Game-Master grid simulation")
    parser.add_argument("--seed", type=int, help="reproduce a specific game scenario")
    parser.add_argument(
        "--policy",
        choices=("baseline", "neural"),
        default="baseline",
        help="use the rule-based baseline or a trained neural policy",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="neural checkpoint path (required for --policy neural)",
    )
    args = parser.parse_args()
    raise SystemExit(
        0
        if run_game(
            seed=args.seed,
            policy_kind=args.policy,
            model_path=args.model,
        )
        else 1
    )
