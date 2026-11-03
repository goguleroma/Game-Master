"""
Adapters package for Game Master.

Adapters provide game-specific implementations of core protocols:
- ActionExecutor
- ObservationAdapter  
- DecisionPolicy
- AgentMemory
"""

from typing import Dict, Type, Any

from core.engine import ActionExecutor, DecisionPolicy, ObservationAdapter, AgentMemory


class AdapterRegistry:
    """Registry for game adapters."""
    
    _adapters: Dict[str, Dict[str, Type]] = {}
    
    @classmethod
    def register(cls, game_id: str, **components: Type) -> None:
        """Register adapter components for a game."""
        cls._adapters[game_id] = components
    
    @classmethod
    def get(cls, game_id: str) -> Dict[str, Type]:
        """Get adapter components for a game."""
        return cls._adapters.get(game_id, {})
    
    @classmethod
    def create_engine(cls, game_id: str, **kwargs: Any) -> Any:
        """Create a GameEngine with the registered adapter for a game."""
        from core.engine import GameEngine
        components = cls.get(game_id)
        if not components:
            raise ValueError(f"No adapter registered for game: {game_id}")
        
        # Extract required components
        policy = components.get("policy")
        input_adapter = components.get("input_adapter")
        memory = components.get("memory")
        observation_adapter = components.get("observation_adapter")
        
        if not policy or not input_adapter:
            raise ValueError(f"Adapter for {game_id} missing required components")
        
        return GameEngine(
            policy=policy,
            input_adapter=input_adapter,
            memory=memory,
            state_extractor=observation_adapter,
        )


# Register grid_30x30 adapter
try:
    from .grid_30x30 import (
        GridGame,
        GridGameInputAdapter,
        GridObservationAdapter,
        GridExplorationPolicy,
        GridMemoryStore,
        GridWorldModel,
        GridPlanner,
        GridPathTreePlanner,
        GridMemoryConsolidator,
        GridNeuralPolicy,
        GridActionManager,
        GridStrategyParameters,
        GridGameConfig,
    )
    
    AdapterRegistry.register(
        "grid_30x30",
        game=GridGame,
        input_adapter=GridGameInputAdapter,
        observation_adapter=GridObservationAdapter,
        policy=GridExplorationPolicy,
        memory=GridMemoryStore,
        world_model=GridWorldModel,
        planner=GridPlanner,
        path_tree_planner=GridPathTreePlanner,
        memory_consolidator=GridMemoryConsolidator,
        neural_policy=GridNeuralPolicy,
        action_manager=GridActionManager,
        strategy=GridStrategyParameters,
        config=GridGameConfig,
    )
except ImportError:
    pass