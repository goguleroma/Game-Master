#!/usr/bin/env python3
"""
Visual Demo - Watch Game Master play in the terminal!
Shows ASCII visualization of the grid, agent position, target, obstacles, and decisions.
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


def render_grid(game, step, action_name, result, reward, show_vision=True):
    """Render the grid as ASCII art."""
    hud = game.observe().get('hud', {})
    grid_info = hud.get('grid', {})
    
    width = grid_info.get('width', 30)
    height = grid_info.get('height', 30)
    pos = grid_info.get('position', [0, 0])
    target = grid_info.get('target')
    obstacles = grid_info.get('obstacles', [])
    entities = grid_info.get('entities', [])
    visible_cells = set(tuple(c) for c in grid_info.get('visible_cells', []))
    
    # Limit display to 30x30 area around agent
    px, py = pos
    view_radius = 15
    min_x, max_x = max(0, px - view_radius), min(width - 1, px + view_radius)
    min_y, max_y = max(0, py - view_radius), min(height - 1, py + view_radius)
    
    lines = []
    border = "+" + "=" * ((max_x - min_x + 1) * 2) + "+"
    lines.append(border)
    lines.append(f"| STEP {step:3d} | ACTION: {action_name:12s} | REWARD: {reward:+.2f} |")
    lines.append(f"| POS: ({px:2d},{py:2d}) | TARGET: {str(target):15s} | HEALTH: {hud.get('health', 10)} |")
    lines.append(f"| RESULT: {str(result)[:60]:60s} |")
    lines.append(border)
    
    for y in range(min_y, max_y + 1):
        row = "| "
        for x in range(min_x, max_x + 1):
            cell = (x, y)
            is_visible = cell in visible_cells
            
            if cell == tuple(pos):
                row += "@ "  # Agent
            elif target and cell == tuple(target):
                row += "T "  # Target
            elif cell in [tuple(o) for o in obstacles]:
                row += "# "  # Obstacle
            elif any(e.get('position') == list(cell) for e in entities):
                # Check entity type
                for e in entities:
                    if e.get('position') == list(cell):
                        if e.get('health') is not None:
                            row += "E "  # Enemy
                        elif e.get('type') == 'key':
                            row += "K "  # Key
                        elif e.get('type') == 'door':
                            row += "D "  # Door
                        elif e.get('type') == 'exit':
                            row += "X "  # Exit
                        else:
                            row += "? "
                        break
            elif not is_visible and show_vision:
                row += ". "  # Unknown
            else:
                row += "  "   # Empty visible
        row += "|"
        lines.append(row)
    
    lines.append(border)
    lines.append("")
    lines.append("Legend: @=Agent  T=Target  #=Wall  E=Enemy  K=Key  D=Door  X=Exit  .=Unknown  =Empty")
    
    return "\n".join(lines)


async def run_visual_demo():
    print("=" * 70)
    print("  GAME MASTER - VISUAL DEMO")
    print("  Watch the agent explore, avoid obstacles, and find the target!")
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
            print(render_grid(game, step, action.name, result.details, reward))
            print(f"TOTAL REWARD: {total_reward:.2f} | CONSOLIDATION: STM={len(memory._stm_buffer)} events")
            
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
            await asyncio.sleep(0.1)  # 10 FPS
            
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