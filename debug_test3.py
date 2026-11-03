from types import SimpleNamespace
from simulated_game.agent.memory import GridMemoryStore
from simulated_game.agent.control_learner import ControlLearner

import tempfile
import os

tmpdir = tempfile.mkdtemp()
database_path = os.path.join(tmpdir, 'memory.sqlite3')

memory = GridMemoryStore(game_id='grid-game', database_path=database_path)
learner = ControlLearner()
state = SimpleNamespace(
    current_position=(1, 1),
    target_position=(100, 1),
    grid_size=(200, 3),
    entities=[
        SimpleNamespace(
            entity_id='entity-1',
            position=(2, 1),
            metadata={},
        )
    ],
)
health = 10
for i in range(3):
    print(f"\n=== Step {i} ===")
    chosen = learner.choose(
        state,
        {'hud': {'health': health}},
        ['control_1', 'control_2'],
        memory,
    )
    print(f"chosen={chosen}")
    print(f"  before pos={state.current_position}, health={health}")
    print(f"  context={learner._context_from_state(state)}")
    print(f"  _pending={learner._pending}")
    
    if chosen == 'control_2':
        state.current_position = (state.current_position[0] + 1, 1)
        health += 1
    else:
        state.current_position = (state.current_position[0] - 1, 1)
        health -= 1
    state.entities[0].position = (state.current_position[0] + 1, 1)
    
    print(f"  after pos={state.current_position}, health={health}")

# Check context models in detail
for ctx, ctm in learner.context_models.items():
    print(f'\nContext {ctx}:')
    for control, atm in ctm.action_models.items():
        print(f'  {control}: samples={len(atm.samples)}')
        for j, s in enumerate(atm.samples):
            print(f'    sample {j}: {s}')
        print(f'    mean_effects: {atm.mean_effects}')