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

# Add some debug to the path_tree
import simulated_game.agent.path_tree as path_tree_module
original_simulate = path_tree_module.PathTreePlanner._simulate

def debug_simulate(self, node, memory):
    result = original_simulate(self, node, memory)
    # Print some info about the root belief
    if node.depth == 0:
        belief = node.belief
        print(f"  Root belief: goal_particles={len(belief.goal_particles)}, entity_models={list(belief.entity_models.keys())}")
        for eid, model in belief.entity_models.items():
            print(f"    Entity {eid}: pos={model.position}, category={getattr(model, 'category', 'unknown')}")
    return result

path_tree_module.PathTreePlanner._simulate = debug_simulate

won = False
caught = False
alerted = False
control_trials = 0

for step in range(200):
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
print(f"Player pos: {game.position}, Enemy pos: {game.enemy_position}")
print(f"Player health: {game.player_health}, Enemy health: {game.enemy_health}")