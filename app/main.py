import argparse
import sys

from core.engine import GameEngine


def run_demo() -> None:
    scenarios = [
        (
            "Menu with start button",
            {
                "objects": ["start_button"],
                "hud": {"status": "menu"},
                "goal": "start",
                "confidence": 0.95,
            },
        ),
        (
            "Movement goal",
            {
                "objects": ["player"],
                "hud": {"status": "playing"},
                "goal": "move_left",
                "confidence": 0.9,
            },
        ),
        (
            "Unseen mixed signals",
            {
                "objects": ["player", "start_button", "key", "hazard"],
                "hud": {"status": "playing", "health": 1},
                "goal": "move_left",
                "confidence": 0.55,
            },
        ),
        (
            "Unfamiliar dungeon situation",
            {
                "objects": ["locked_gate", "glowing_rune", "patrolling_guard"],
                "hud": {"area": "dungeon", "health": 20, "noise": "high"},
                "goal": "escape_without_being_seen",
                "confidence": 0.72,
            },
        ),
        (
            "Uncertain scene",
            {
                "objects": ["unknown"],
                "hud": {"status": "unclear"},
                "goal": None,
                "confidence": 0.1,
            },
        ),
        ("No observation data", None),
    ]

    engine = GameEngine()
    print(
        "DEMO SIMULATA - nessun gioco o input reale viene controllato.",
        file=sys.stderr,
        flush=True,
    )
    for name, frame in scenarios:
        print(f"\n--- {name} ---", file=sys.stderr, flush=True)
        engine.process_frame(frame, source="demo")

    print(
        f"\nEsperienze registrate in memoria: {len(engine.memory.records)}",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Game-Master prototype")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="show a simulated observation/decision/action trace",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        GameEngine().run()


if __name__ == "__main__":
    main()
