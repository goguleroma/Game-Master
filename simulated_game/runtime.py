from core.engine import ActionExecutor, DecisionPolicy, GameEngine
from adapters.grid_30x30 import (
    GridExplorationPolicy,
    GridMemoryStore,
    GridStrategyParameters,
    GridGame,
    GridGameInputAdapter,
    GridObservationAdapter,
)


def create_grid_engine(
    game: GridGame,
    memory: GridMemoryStore | None = None,
    input_adapter: ActionExecutor | None = None,
    policy: DecisionPolicy | None = None,
    strategy: GridStrategyParameters | None = None,
) -> GameEngine:
    agent_policy = (
        policy
        if policy is not None
        else GridExplorationPolicy(strategy)
    )
    game_memory = (
        memory
        if memory is not None
        else GridMemoryStore(game_id="grid_30x30_v1")
    )
    game_adapter = (
        input_adapter
        if input_adapter is not None
        else GridGameInputAdapter(game)
    )
    observation_adapter = GridObservationAdapter()
    # GameEngine expects: policy, input_adapter, observation_adapter, memory
    return GameEngine(agent_policy, game_adapter, observation_adapter, game_memory)