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

# Patch the observe_entity to print debug info
original_observe_entity = engine.policy.control_learner.observe_entity

def debug_observe_entity(self, entity_id, position, agent_position, turn, metadata=None):
    print(f"  observe_entity: {entity_id} at {position}, agent at {agent_position}, turn={turn}")
    return original_observe_entity(entity_id, position, agent_position, turn, metadata)

import simulated_game.agent.control_learner as cl_module
cl_module.WorldModelLearner.observe_entity = debug_observe_entity

won = False
caught = False
alerted = False
control_trials = 0

for step in range(200):
    print(f"\n=== Step {step} ===")
    engine.process_frame(game.observe(), source="test_seeded_exploration")
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