import torch
from simulated_game.agent.control_learner import ControlLearner, WorldModelLearner
from simulated_game.agent.memory import GridMemoryStore
from simulated_game.game import GridGame
from simulated_game.agent.exploration_policy import ExplorationPolicy
from simulated_game.runtime import create_grid_engine
from simulated_game.agent.planner import GridStrategyParameters

device = torch.device('cpu')

# Test ExplorationPolicy directly
policy = ExplorationPolicy(GridStrategyParameters())
memory = GridMemoryStore()

game = GridGame(seed=15, moving_target=True, pursuer_enabled=True)

print("Testing ExplorationPolicy directly:")
for turns in range(1, 4):
    obs = game.observe()
    print(f"\nTurn {turns}: pos={game.position}")
    print(f"  Controls: {obs['controls']}")
    print(f"  Entities: {obs['hud']['grid'].get('entities', [])}")
    
    action = policy.decide(obs, memory)
    print(f"  Action returned: {action}")
    
    if action is None:
        print("  Action is None!")
        continue
        
    if action.name in obs['controls']:
        print(f"  -> CONTROL ACTION: {action.name}")
    else:
        print(f"  -> CARDINAL ACTION: {action.name}")
    
    result = game.execute(action)
    print(f"  Status: {result.details.get('status')}")
    if result.details.get('status') in {'goal_reached', 'won', 'caught', 'lost', 'waiting_for_visible_target'}:
        break

# Now test control_learner directly
print("\n\nTesting ControlLearner directly:")
learner = ControlLearner()
state = policy.world_model.update(obs)

# Get context from state
context = WorldModelLearner._context_from_state(state)
print(f"Context from state: {context}")

# Check what choose returns
available_controls = obs['controls']
desired_direction = (1, 0)  # move_right

print(f"Available controls: {available_controls}")
print(f"Desired direction: {desired_direction}")

control = learner.choose(state, obs, available_controls, memory, desired_direction)
print(f"Control chosen: {control}")

if control:
    pred = learner.predicted_displacement(control, context)
    print(f"Predicted displacement: {pred}")