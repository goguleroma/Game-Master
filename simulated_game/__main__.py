import argparse

from simulated_game.runner import run_game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Game-Master grid simulation")
    parser.add_argument("--seed", type=int, help="reproduce a specific game scenario")
    args = parser.parse_args()
    raise SystemExit(0 if run_game(seed=args.seed) else 1)
