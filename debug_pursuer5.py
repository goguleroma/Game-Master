from simulated_game.agent.exploration_policy import ExplorationPolicy
from simulated_game.agent.memory import GridMemoryStore as MemoryStore
from simulated_game.game import GridGame
from simulated_game.runner import create_grid_engine
from core.types import Action

# Test with a single seed
seed = 0
game = GridGame(seed=seed)
engine = create_grid_engine(game)
engine.logger.logger.disabled = True

won = False
caught = False
alerted = False
control_trials = 0

for step in range(200):
    obs = game.observe()
    print(f"Step {step}: agent_pos={game.position}, target_pos={game.target}, enemy_pos={game.enemy_position}, enemy_alerted={game.enemy_alerted}")
    engine.process_frame(obs, source="test_seeded_exploration")
    if game.won:
        won = True
        print(f"Step {step}: WON!")
        break
    if game.caught:
        caught = True
        print(f"Step {step}: CAUGHT!")
        break
    if game.enemy_alerted:
        alerted = True
        print(f"Step {step}: ENEMY ALERTED!")
    control_trials = sum(
        experience["action"].name.startswith("control_")
        for experience in engine.memory.experiences
    )

print(f"Result: won={won}, caught={caught}, alerted={alerted}, control_trials={control_trials}")