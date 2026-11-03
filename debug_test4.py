from types import SimpleNamespace
from simulated_game.agent.control_learner import ControlLearner

learner = ControlLearner()
state = SimpleNamespace(current_position=(0, 0), entities=[])
controls = ["opaque_a", "opaque_b"]

print("Step 1:")
first = learner.choose(
    state,
    {"position": list(state.current_position)},
    controls,
    desired_direction=(1, 0),
)
print(f"  first={first}")
print(f"  context={learner._context_from_state(state)}")
print(f"  _pending={learner._pending}")
state.current_position = (-1, 0)

print("\nStep 2:")
second = learner.choose(
    state,
    {"position": list(state.current_position)},
    controls,
    desired_direction=(1, 0),
)
print(f"  second={second}")
print(f"  context={learner._context_from_state(state)}")
print(f"  _pending={learner._pending}")
state.current_position = (0, 0)

print("\nStep 3:")
third = learner.choose(
    state,
    {"position": list(state.current_position)},
    controls,
    desired_direction=(1, 0),
)
print(f"  third={third}")
print(f"  context={learner._context_from_state(state)}")

# Check context models
for ctx, ctm in learner.context_models.items():
    print(f'\nContext {ctx}:')
    for control, atm in ctm.action_models.items():
        print(f'  {control}: samples={len(atm.samples)}')
        for j, s in enumerate(atm.samples):
            print(f'    sample {j}: {s}')
        print(f'    mean_effects: {atm.mean_effects}')

print(f"\nlatest_inference: {learner.latest_inference}")
print(f"predicted_displacement(opaque_a): {learner.predicted_displacement('opaque_a')}")
print(f"predicted_displacement(opaque_b): {learner.predicted_displacement('opaque_b')}")