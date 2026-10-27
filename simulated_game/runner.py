import sys

from core.engine import GameEngine
from simulated_game.game import GridGame, GridGameInputAdapter


def run_game(max_steps: int = 20) -> bool:
    game = GridGame()
    engine = GameEngine()
    engine.input_adapter = GridGameInputAdapter(game)

    print(
        "Game-Master grid test (simulato, nessun input reale).",
        file=sys.stderr,
        flush=True,
    )
    for step in range(1, max_steps + 1):
        frame = game.observe()
        print(
            f"\nStep {step}: posizione={game.position}, "
            f"obiettivo={'visibile' if frame['hud']['grid']['target'] is not None else 'fuori visuale'}",
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
        if result is not None and result.details.get("status") == "waiting_for_visible_target":
            print(
                "\nGM non vede l'obiettivo e sceglie wait: non ha ancora una strategia di esplorazione.",
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
            elif position in game.obstacles:
                row.append("#")
            else:
                row.append(".")
        rows.append(" ".join(row))
    return "\n".join(rows)
