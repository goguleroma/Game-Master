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

# Patch the control_learner to debug risk computation
original_compute_risk = engine.policy.control_learner._compute_entity_risk_at

def debug_compute_risk(self, agent_pos, desired_direction, context, available_controls):
    result = original_compute_risk(agent_pos, desired_direction, context, available_controls)
    if result:
        print(f"  RISK: agent_pos={agent_pos}, dir={desired_direction}, context={context}, risk={result}")
        for eid, model in self.entity_models.items():
            print(f"    Entity {eid}: pos={model.positions[-1][0] if model.positions else 'none'}, category={model.category}, conf={model.category_confidence:.2f}, approach_streak={model.approach_streak}")
    return result

import simulated_game.agent.control_learner as cl_module
cl_module.WorldModelLearner._compute_entity_risk_at = debug_compute_risk

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