#!/usr/bin/env python3
"""
Visual Demo - Watch Game Master play in the terminal!
Shows the FULL GRID so you can see walls are STATIC.
"""

import asyncio
import time
import os
import sys
sys.path.insert(0, r'C:/Users/0102/Documents/GitHub/Game-Master')

from simulated_game.agent.memory import GridMemoryStore, Scope
from simulated_game.agent.memory_consolidator import ConsolidationConfig
from simulated_game.game import GridGame
from simulated_game.agent.exploration_policy import ExplorationPolicy
from simulated_game.agent.planner import GridStrategyParameters
from core.types import Action


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def render_full_grid(game, step, action_name, result, reward):
    """Render the ENTIRE 30x30 grid so walls don't appear to move."""
    hud = game.observe().get('hud', {})
    grid_info = hud.get('grid', {})
    
    width = grid_info.get('width', 30)
    height = grid_info.get('height', 30)
    pos = grid_info.get('position', [0, 0])
    target = grid_info.get('target')
    obstacles = grid_info.get('obstacles', [])
    entities = grid_info.get('entities', [])
    visible_cells = set(tuple(c) for c in grid_info.get('visible_cells', []))
    
    # Build full static grid once
    static_grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Place static obstacles
    for ox, oy in obstacles:
        static_grid[oy][ox] = '#'
    
    # Place static entities
    for e in entities:
        if e.get('type') == 'key':
            ex, ey = e.get('position', [0, 0])
            static_grid[ey][ex] = 'K'
        elif e.get('type') == 'door':
            ex, ey = e.get('position', [0, 0])
            static_grid[ey][ex] = 'D'
        elif e.get('type') == 'exit':
            ex, ey = e.get('position', [0, 0])
            static_grid[ey][ex] = 'X'
        elif e.get('health') is not None:
            ex, ey = e.get('position', [0, 0])
            static_grid[ey][ex] = 'E'
    
    # Place target if visible
    if target:
        tx, ty = target
        static_grid[ty][tx] = 'T'
    
    lines = []
    lines.append("=" * 70)
    lines.append(f" GAME MASTER - FULL GRID VIEW (30x30) - Walls are STATIC!")
    lines.append("=" * 70)
    lines.append(f" Step: {step:3d} | Action: {action_name:12s} | Reward: {reward:+.2f} | Total: {getattr(render_full_grid, 'total_reward', 0):.2f}")
    lines.append(f" Pos: ({pos[0]:2d},{pos[1]:2d}) | Target: {str(target):15s} | Health: {hud.get('health', 10)}")
    lines.append(f" Result: {str(result)[:60]}")
    lines.append("-" * 70)
    
    # Show entire grid
    for y in range(height):
        row = ""
        for x in range(width):
            cell = (x, y)
            is_visible = cell in visible_cells
            
            if cell == tuple(pos):
                row += "@"  # Agent
            elif is_visible:
                ch = static_grid[y][x]
                if ch == ' ':
                    row += "."
                else:
                    row += ch
            else:
                row += " "  # Unknown = blank
        lines.append(f" {y:2d} |{row}|")
    
    lines.append("-" * 70)
    lines.append("    " + "".join(str(i % 10) for i in range(width)))
    lines.append("")
    lines.append("Legend: @=Agent  T=Target  #=Wall  E=Enemy  K=Key  D=Door  X=Exit  .=VisibleEmpty  =Unknown")
    lines.append("WALLS ARE STATIC - they don't move! The viewport just follows the agent.")
    
    return "\n".join(lines)


async def run_visual_demo():
    print("=" * 70)
    print("  GAME MASTER - FULL GRID VISUAL DEMO")
    print("  Shows the ENTIRE 30x30 grid - walls are STATIC!")
    print("=" * 70)
    print()
    print("Press Ctrl+C to stop")
    print()
    time.sleep(2)
    
    # Create memory with consolidation
    memory = GridMemoryStore(
        game_id='visual_demo',
        enable_consolidation=True,
        consolidation_config=ConsolidationConfig(
            stm_significance_threshold=0.05,
            stm_min_reward_magnitude=0.01,
            stm_novelty_threshold=0.05,
            stm_recurrence_threshold=1,
            mtm_confidence_threshold=0.2,
            mtm_confirmation_threshold=1,
            ltm_pattern_min_support=2,
            ltm_pattern_confidence=0.2,
        )
    )
    
    # Create game
    game = GridGame(seed=42, moving_target=True, pursuer_enabled=True)
    policy = ExplorationPolicy(GridStrategyParameters())
    policy.set_memory(memory)
    
    obs = game.observe()
    step = 0
    total_reward = 0
    won = False
    
    try:
        while not won and step < 200:
            obs = game.observe()
            
            if obs.get('hud', {}).get('status') != 'playing':
                break
            
            action = policy.decide(obs, memory)
            if action is None:
                available = game.get_available_actions()
                if available:
                    action = Action(name=available[0])
                else:
                    break
            
            result = game.execute(action)
            
            # Check win
            next_obs = game.observe()
            won_now = 'won' in str(result.details) or next_obs.get('hud', {}).get('status') == 'won'
            
            # Reward
            if result.success:
                if won_now:
                    reward = 10.0
                elif 'moved' in str(result.details):
                    reward = 0.2
                else:
                    reward = 0.0
            else:
                reward = -0.05
            
            total_reward += reward
            render_full_grid.total_reward = total_reward
            
            memory.record_experience({
                'type': 'step',
                'game': 'visual_demo',
                'episode': 0,
                'step': step,
                'obs': obs,
                'action': action.name,
                'reward': reward,
                'result': str(result.details)
            })
            
            # Render
            clear_screen()
            print(render_full_grid(game, step, action.name, result.details, reward))
            
            if memory._consolidator:
                stats = memory._consolidator.get_stats()
                print(f"  STM->MTM: {stats.get('stm_to_mtm', 0)} | MTM->LTM: {stats.get('mtm_to_ltm', 0)} | LTM->GSM: {stats.get('ltm_to_gsm', 0)}")
            
            if won_now:
                print("\n***  AGENT WON!  ***")
                memory.add_knowledge(
                    key='win_strategy_visual',
                    value={'final_pos': next_obs.get('hud', {}).get('grid', {}).get('position'), 'steps': step},
                    confidence=0.9,
                    context={'game': 'visual_demo', 'steps': step},
                    scope=Scope.GAME,
                    transferable=True
                )
                break
            
            step += 1
            await asyncio.sleep(0.08)  # ~12 FPS
            
            # Run consolidation every 20 steps
            if step % 20 == 0 and memory._consolidator:
                await memory.consolidator.consolidate_step()
                
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    
    print("\n" + "=" * 70)
    print("FINAL STATS")
    print("=" * 70)
    print(f"Steps: {step}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Won: {'YES!' if won else 'No'}")
    
    stats = memory.get_consolidation_stats()
    print(f"\nConsolidation: {stats}")
    
    print("\nKnowledge (GAME scope):")
    for item in memory.get_valid_knowledge(scope=Scope.GAME):
        print(f"  {item.key}: {item.value} (conf={item.confidence:.2f})")


if __name__ == '__main__':
    asyncio.run(run_visual_demo())