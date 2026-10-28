import sys
from pathlib import Path

from simulated_game.agent.memory import GridMemoryStore
from simulated_game.game import GridGame
from simulated_game.runtime import create_grid_engine
from simulated_game.strategy_config import load_strategy


def run_game(
    max_steps: int = 200,
    seed: int | None = None,
    policy_kind: str = "baseline",
    model_path: Path | None = None,
) -> bool:
    game = GridGame(seed=seed)
    if policy_kind == "baseline":
        from simulated_game.agent.exploration_policy import ExplorationPolicy

        policy = ExplorationPolicy(
            load_strategy(Path(".gm-strategy-dynamic.json"), "dynamic")
        )
    elif policy_kind == "neural":
        from simulated_game.agent.neural_policy import NeuralGridPolicy

        if model_path is None or not model_path.is_file():
            raise FileNotFoundError(
                "A trained neural model is required; run "
                "python -m simulated_game.neural_experiment first."
            )
        policy = NeuralGridPolicy(seed=seed or 0)
        policy.agent.load(model_path, resume_training=False)
    else:
        raise ValueError(f"Unknown policy: {policy_kind}")
    engine = create_grid_engine(
        game,
        memory=GridMemoryStore(
            game_id="grid_30x30_v1",
            database_path=Path(".gm-memory.sqlite3"),
        ),
        policy=policy,
    )

    print(
        f"Game-Master grid test (simulato, policy={policy_kind}, seed={seed}, "
        "nessun input reale).",
        file=sys.stderr,
        flush=True,
    )
    for step in range(1, max_steps + 1):
        frame = game.observe()
        print(
            f"\nStep {step}: posizione={game.position}, "
            f"PV={game.player_health}, "
            f"entita={game.enemy_health if game.enemy_position is not None else 'rimossa'}, "
            f"obiettivo={'visibile' if frame['hud']['grid']['target'] is not None else 'fuori visuale'}, "
            f"nemico={'inseguimento' if game.enemy_alerted else 'inattivo'}",
            file=sys.stderr,
            flush=True,
        )
        print(render(game), file=sys.stderr, flush=True)
        result = engine.process_frame(frame, source="grid_game")
        if game.won:
            print(f"\nObiettivo raggiunto in {step} mosse.", file=sys.stderr, flush=True)
            print(
                f"Esperienze registrate: {len(engine.memory.records)}",
                file=sys.stderr,
                flush=True,
            )
            return True
        if game.caught:
            print(f"\nGM sconfitto dopo {step} turni.", file=sys.stderr, flush=True)
            print(
                f"Esperienze registrate: {len(engine.memory.records)}",
                file=sys.stderr,
                flush=True,
            )
            return False
        if result is not None and result.details.get("status") == "waiting_for_visible_target":
            print(
                "\nGM non trova una mossa utile verso il traguardo o una frontiera esplorabile.",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"Esperienze registrate: {len(engine.memory.records)}",
                file=sys.stderr,
                flush=True,
            )
            return False

    print(
        f"\nObiettivo non raggiunto in {max_steps} mosse.",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"Esperienze registrate: {len(engine.memory.records)}",
        file=sys.stderr,
        flush=True,
    )
    return False


def render(game: GridGame) -> str:
    rows = []
    for y in range(game.height):
        row = []
        for x in range(game.width):
            position = (x, y)
            if not game.is_visible(position):
                row.append("?")
            elif position == game.position:
                row.append("P")
            elif position == game.target:
                row.append("G")
            elif position == game.enemy_position:
                row.append("E")
            elif position in game.obstacles:
                row.append("#")
            else:
                row.append(".")
        rows.append(" ".join(row))
    return "\n".join(rows)
