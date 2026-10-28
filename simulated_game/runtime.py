from core.engine import ActionExecutor, DecisionPolicy, GameEngine
from simulated_game.agent import ExplorationPolicy
from simulated_game.agent.memory import GridMemoryStore
from simulated_game.agent.planner import GridStrategyParameters
from simulated_game.game import GridGame, GridGameInputAdapter


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
        else ExplorationPolicy(strategy)
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
    return GameEngine(agent_policy, game_adapter, game_memory)
