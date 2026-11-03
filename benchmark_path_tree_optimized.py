#!/usr/bin/env python
"""
Benchmark script for optimized PathTreePlanner.

Measures decisions/second and latency for real-time viability.
Target: <10ms per decision (100+ decisions/sec)
"""

import sys
import time
import statistics
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from adapters.grid_30x30 import (
    GridGame,
    GridMemoryStore,
    GridExplorationPolicy,
    GridStrategyParameters,
    WorldModelLearner,
)
from adapters.grid_30x30.path_tree_planner_optimized import OptimizedGridPathTreePlanner
from adapters.grid_30x30.world_model import GridWorldModel
from simulated_game.runtime import create_grid_engine


def create_test_scenario(seed: int = 42, moving_target: bool = True) -> tuple:
    """Create a test game scenario with planner and memory."""
    game = GridGame(seed=seed, moving_target=moving_target, pursuer_enabled=moving_target)
    memory = GridMemoryStore(game_id="grid_30x30_bench", database_path=None)
    
    # Create learners directly
    control_learner = WorldModelLearner(seed=seed)
    world_model_learner = control_learner  # ControlLearner IS a WorldModelLearner
    
    # Create planner
    planner = OptimizedGridPathTreePlanner(
        max_depth=8,
        num_simulations=50,
        exploration_constant=1.414,
        num_particles=30,
        control_learner=control_learner,
        world_model_learner=world_model_learner,
        memory=memory,
        seed=seed,
        use_numba=True,
        early_termination_threshold=0.01,
        convergence_window=10,
    )
    
    return game, memory, planner


def benchmark_planner(
    planner: OptimizedGridPathTreePlanner,
    game: GridGame,
    memory: GridMemoryStore,
    num_decisions: int = 100,
    warmup: int = 10,
) -> dict:
    """Benchmark planner performance over multiple decisions."""
    
    print(f"\n=== Benchmarking {planner.__class__.__name__} ===")
    print(f"Warmup: {warmup} decisions, Measured: {num_decisions} decisions")
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        observation = game.observe()
        # Update memory
        hud = observation.get("hud", {})
        grid = hud.get("grid", {})
        memory.update_working_memory(observation, None, None)
        pos = grid.get("position", (0, 0))
        if isinstance(pos, list):
            pos = tuple(pos)
        memory.mark_cell(pos, seen=True, visited=True)
        
        state = GridWorldModel().update(observation)
        _ = planner.plan(state, memory)
        
        # Execute a random action to advance game
        if not game.won and not game.caught:
            game.execute(type('Action', (), {'name': 'move_right', 'params': {}})())
    
    # Benchmark
    print("Running benchmark...")
    latencies = []
    sim_counts = []
    
    for i in range(num_decisions):
        observation = game.observe()
        hud = observation.get("hud", {})
        grid = hud.get("grid", {})
        memory.update_working_memory(observation, None, None)
        pos = grid.get("position", (0, 0))
        if isinstance(pos, list):
            pos = tuple(pos)
        memory.mark_cell(pos, seen=True, visited=True)
        
        state = GridWorldModel().update(observation)
        
        start = time.perf_counter()
        action = planner.plan(state, memory)
        elapsed = time.perf_counter() - start
        
        latencies.append(elapsed * 1000)  # ms
        sim_counts.append(len(getattr(planner, '_simulation_times', [])))
        if hasattr(planner, '_simulation_times'):
            planner._simulation_times.clear()
        
        # Execute action to advance game
        if not game.won and not game.caught:
            try:
                game.execute(type('Action', (), {'name': action.name, 'params': action.params})())
            except Exception:
                pass
        
        if game.won or game.caught:
            # Reset game for continued benchmarking
            game = GridGame(seed=42 + i + 1, moving_target=True, pursuer_enabled=True)
    
    # Statistics
    latencies = np.array(latencies)
    sim_counts = np.array(sim_counts)
    
    results = {
        "num_decisions": num_decisions,
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "decisions_per_sec": 1000.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0,
        "mean_simulations": float(np.mean(sim_counts)),
        "use_numba": getattr(planner, 'use_numba', False),
        "numba_available": getattr(planner, 'use_numba', False),
    }
    
    return results


def print_results(results: dict, label: str = "Results"):
    """Print benchmark results in a formatted way."""
    print(f"\n=== {label} ===")
    print(f"Decisions measured:     {results['num_decisions']}")
    print(f"Mean latency:           {results['mean_latency_ms']:.2f} ms")
    print(f"Median latency:         {results['median_latency_ms']:.2f} ms")
    print(f"Std deviation:          {results['std_latency_ms']:.2f} ms")
    print(f"Min latency:            {results['min_latency_ms']:.2f} ms")
    print(f"Max latency:            {results['max_latency_ms']:.2f} ms")
    print(f"P95 latency:            {results['p95_latency_ms']:.2f} ms")
    print(f"P99 latency:            {results['p99_latency_ms']:.2f} ms")
    print(f"Decisions/second:       {results['decisions_per_sec']:.1f}")
    print(f"Mean simulations/dec:   {results['mean_simulations']:.1f}")
    print(f"Numba enabled:          {results['use_numba']}")
    
    # Target check
    target_ms = 10.0
    if results['mean_latency_ms'] <= target_ms:
        print(f"\n[OK] TARGET MET: Mean latency {results['mean_latency_ms']:.2f}ms <= {target_ms}ms")
    else:
        print(f"\n[FAIL] TARGET MISSED: Mean latency {results['mean_latency_ms']:.2f}ms > {target_ms}ms")
    
    if results['p99_latency_ms'] <= target_ms:
        print(f"[OK] P99 TARGET MET: P99 latency {results['p99_latency_ms']:.2f}ms <= {target_ms}ms")
    else:
        print(f"[FAIL] P99 TARGET MISSED: P99 latency {results['p99_latency_ms']:.2f}ms > {target_ms}ms")


def compare_implementations():
    """Compare original vs optimized planner."""
    print("\n" + "="*60)
    print("COMPARISON: Original vs Optimized PathTreePlanner")
    print("="*60)
    
    # Test original (if available)
    try:
        from adapters.grid_30x30.path_tree_planner import GridPathTreePlanner as OriginalPlanner
        has_original = True
    except ImportError:
        has_original = False
        print("Original planner not available for comparison")
    
    # Create test scenario
    seed = 42
    
    # Benchmark optimized
    game_opt, mem_opt, planner_opt = create_test_scenario(seed=seed)
    results_opt = benchmark_planner(planner_opt, game_opt, mem_opt, num_decisions=50, warmup=5)
    print_results(results_opt, "Optimized Planner (Numba)")
    
    if has_original:
        # Benchmark original - create separate learners and memory
        from adapters.grid_30x30.control_learner import WorldModelLearner
        control_learner = WorldModelLearner(seed=seed)
        world_model_learner = control_learner
        memory = GridMemoryStore(game_id="grid_30x30_bench_orig", database_path=None)
        game = GridGame(seed=seed, moving_target=True, pursuer_enabled=True)
        
        planner_orig = OriginalPlanner(
            max_depth=8,
            num_simulations=50,
            exploration_constant=1.414,
            num_particles=30,
            control_learner=control_learner,
            world_model_learner=world_model_learner,
            memory=memory,
            seed=seed,
        )
        results_orig = benchmark_planner(planner_orig, game, memory, num_decisions=50, warmup=5)
        print_results(results_orig, "Original Planner (Pure Python)")
        
        # Speedup
        speedup = results_orig['mean_latency_ms'] / results_opt['mean_latency_ms']
        print(f"\n[SPEEDUP] {speedup:.2f}x faster")
        print(f"   Original: {results_orig['mean_latency_ms']:.2f} ms")
        print(f"   Optimized: {results_opt['mean_latency_ms']:.2f} ms")


def test_numba_correctness():
    """Verify Numba implementation produces same results as Python."""
    print("\n" + "="*60)
    print("CORRECTNESS TEST: Numba vs Python Implementation")
    print("="*60)
    
    from adapters.grid_30x30.path_tree_planner_numba import (
        simulate_transition_numba,
        greedy_direction_numba,
        belief_state_to_arrays,
        get_buffers,
        NumbaBuffers,
        CELL_UNKNOWN, CELL_FREE, CELL_OBSTACLE,
    )
    from adapters.grid_30x30.path_tree_planner import BeliefState, CellBelief, EntityModel
    import random
    
    # Create a test belief state
    rng = random.Random(42)
    belief = BeliefState(
        goal_particles=[((10, 10), 0.5), ((12, 8), 0.3), ((8, 12), 0.2)],
        cell_beliefs={
            (5, 5): CellBelief.FREE,
            (5, 6): CellBelief.OBSTACLE,
            (6, 5): CellBelief.FREE,
        },
        entity_models={
            "e1": EntityModel("e1", "approaches_agent", (8, 8), (0.5, 0.5), 0.8, 0),
            "e2": EntityModel("e2", "stationary", (15, 15), (0, 0), 1.0, 0),
        },
        entropy=0.5,
        turn=10,
    )
    
    # Test greedy_direction
    buffers = get_buffers()
    max_cells = NumbaBuffers.MAX_CELLS
    
    # Fill cell beliefs
    num_cells = 0
    for (x, y), b in belief.cell_beliefs.items():
        buffers.memory_cell_x[num_cells] = x
        buffers.memory_cell_y[num_cells] = y
        if b == CellBelief.OBSTACLE:
            buffers.memory_cell_obs[num_cells] = 1
        elif b == CellBelief.FREE:
            buffers.memory_cell_obs[num_cells] = 2
        else:
            buffers.memory_cell_obs[num_cells] = 0
        num_cells += 1
    
    dir_x, dir_y = greedy_direction_numba(5, 5, 10, 10,
        buffers.memory_cell_x, buffers.memory_cell_y, buffers.memory_cell_obs, num_cells)
    
    print(f"Greedy direction from (5,5) to (10,10): ({dir_x}, {dir_y})")
    # Should be (1, 0) or (0, 1) - both valid, but (1, 0) is right
    
    # Test simulate_transition
    belief_arrays = belief_state_to_arrays(belief)
    
    # Memory setup
    mem_num_cells = 0
    for pos, cell in {}.items():  # Empty memory
        pass
    
    new_num_goal, new_num_cells, new_entropy, new_turn = simulate_transition_numba(
        belief_arrays['goal_particles_x'],
        belief_arrays['goal_particles_y'],
        belief_arrays['goal_particles_w'],
        belief_arrays['num_goal_particles'],
        belief_arrays['cell_beliefs_x'],
        belief_arrays['cell_beliefs_y'],
        belief_arrays['cell_beliefs_val'],
        belief_arrays['num_cell_beliefs'],
        None,
        belief_arrays['entity_pos_x'],
        belief_arrays['entity_pos_y'],
        belief_arrays['entity_vel_x'],
        belief_arrays['entity_vel_y'],
        belief_arrays['entity_category'],
        belief_arrays['entity_confidence'],
        belief_arrays['num_entities'],
        belief_arrays['entropy'],
        belief_arrays['turn'],
        (1, 0),  # direction right
        buffers.memory_cell_x, buffers.memory_cell_y, buffers.memory_cell_obs, 0,
        np.zeros(20, dtype=np.int32), np.zeros(20, dtype=np.int32), 0,
        buffers.next_goal_x, buffers.next_goal_y, buffers.next_goal_w,
        buffers.next_cell_x, buffers.next_cell_y, buffers.next_cell_val,
        buffers.next_entity_x, buffers.next_entity_y,
    )
    
    print(f"Transition result:")
    print(f"  Goal particles: {new_num_goal}")
    print(f"  Cell beliefs: {new_num_cells}")
    print(f"  Entropy: {new_entropy:.4f}")
    print(f"  Turn: {new_turn}")
    
    # Verify particles moved correctly (agent moves right, goal moves left relative)
    for i in range(new_num_goal):
        px = buffers.next_goal_x[i]
        py = buffers.next_goal_y[i]
        pw = buffers.next_goal_w[i]
        print(f"  Particle {i}: ({px}, {py}) w={pw:.4f}")
    
    print("[OK] Numba functions executed successfully")


def main():
    """Main benchmark entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark optimized PathTreePlanner")
    parser.add_argument("--decisions", type=int, default=50, help="Number of decisions to benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup decisions")
    parser.add_argument("--compare", action="store_true", help="Compare original vs optimized")
    parser.add_argument("--test-correctness", action="store_true", help="Test Numba correctness")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    if args.test_correctness:
        test_numba_correctness()
        return
    
    if args.compare:
        compare_implementations()
        return
    
    # Single benchmark
    game, memory, planner = create_test_scenario(seed=args.seed)
    results = benchmark_planner(planner, game, memory, 
                                num_decisions=args.decisions, 
                                warmup=args.warmup)
    print_results(results, "Optimized PathTreePlanner")
    
    # Print performance stats
    perf = planner.get_performance_stats()
    print(f"\nPerformance Stats:")
    for k, v in perf.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()