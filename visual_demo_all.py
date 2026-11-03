#!/usr/bin/env python3
"""
Visual Demo - Watch Game Master play ALL 4 games!
Shows the FULL GRID for each game so you can see static elements don't move.
"""

import asyncio
import time
import os
import sys
sys.path.insert(0, r'C:/Users/0102/Documents/GitHub/Game-Master')

from simulated_game.agent.memory import GridMemoryStore, Scope
from simulated_game.agent.memory_consolidator import ConsolidationConfig
from simulated_game.test_games import create_test_game
from simulated_game.agent.exploration_policy import ExplorationPolicy
from simulated_game.agent.planner import GridStrategyParameters
from core.types import Action


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


class GameVisualizer:
    """Base visualizer for grid-based games."""
    
    def __init__(self, game, game_name):
        self.game = game
        self.game_name = game_name
        self.total_reward = 0
        self.step = 0
        
    def render(self, action_name, result, reward):
        raise NotImplementedError


class GridGameVisualizer(GameVisualizer):
    """Visualizer for GridGameV1, GridGameV2, MazeGame - shows full 2D grid."""
    
    def render(self, action_name, result, reward):
        obs = self.game.observe()
        hud = obs.get('hud', {})
        grid_info = hud.get('grid', {}) or hud.get('world', {})
        
        width = grid_info.get('width', 30)
        height = grid_info.get('height', 30)
        pos = grid_info.get('position', [0, 0])
        
        # Extract game-specific data
        if self.game_name == 'maze':
            target = grid_info.get('goal_position') or grid_info.get('target')
            obstacles = []
            # Maze walls are in the grid
            entities = obs.get('objects', [])
            visible_cells = set(tuple(c) for c in grid_info.get('visible_cells', []))
            keys = grid_info.get('keys_collected', 0)
            keys_total = grid_info.get('keys_total', 0)
            doors = grid_info.get('doors', [])
        else:
            target = grid_info.get('target')
            obstacles = grid_info.get('obstacles', [])
            visible_cells = set(tuple(c) for c in grid_info.get('visible_cells', []))
            keys = 0
            keys_total = 0
            doors = []
        
        # Build static grid
        static_grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Place static obstacles/walls
        for ox, oy in obstacles:
            if 0 <= ox < width and 0 <= oy < height:
                static_grid[oy][ox] = '#'
        
        # Place target
        if target:
            tx, ty = target
            if 0 <= tx < width and 0 <= ty < height:
                static_grid[ty][tx] = 'T'
        
        # Build display
        lines = []
        lines.append("=" * 70)
        lines.append(f" {self.game_name.upper()} - FULL GRID ({width}x{height}) - Walls are STATIC!")
        lines.append("=" * 70)
        lines.append(f" Step: {self.step:3d} | Action: {action_name:12s} | Reward: {reward:+.2f} | Total: {self.total_reward:.2f}")
        lines.append(f" Pos: ({pos[0]:2d},{pos[1]:2d}) | Target: {str(target):15s} | Health: {hud.get('health', 10)}")
        if keys > 0:
            lines.append(f" Keys: {keys}/{keys_total}")
        lines.append(f" Result: {str(result)[:60]}")
        lines.append("-" * 70)
        
        px, py = int(pos[0]), int(pos[1])
        
        for y in range(height):
            row = ""
            for x in range(width):
                cell = (x, y)
                is_visible = cell in visible_cells
                
                if cell == (px, py):
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
        lines.append("Legend: @=Agent  T=Target  #=Wall  .=VisibleEmpty  =Unknown")
        lines.append("WALLS ARE STATIC - they don't move! The viewport just follows the agent.")
        
        return "\n".join(lines)


class PlatformerVisualizer(GameVisualizer):
    """Visualizer for PlatformerGame - side view with platforms."""
    
    def render(self, action_name, result, reward):
        obs = self.game.observe()
        hud = obs.get('hud', {})
        world = hud.get('world', {})
        
        width = world.get('width', 80)
        height = world.get('height', 30)
        pos = world.get('position', [0, 0])
        vel = world.get('velocity', [0, 0])
        on_ground = world.get('on_ground', True)
        facing = world.get('facing_right', True)
        platforms = world.get('platforms', [])
        goal = world.get('goal')
        
        px, py = pos[0], pos[1]
        gx, gy = goal if goal else (None, None)
        
        # Build static world (platforms don't move)
        static_grid = [[' ' for _ in range(int(width))] for _ in range(int(height))]
        
        for plat_x, plat_y, plat_w in platforms:
            px_int = int(plat_x)
            py_int = int(plat_y)
            pw_int = int(plat_w)
            for x in range(px_int, min(px_int + pw_int, width)):
                if 0 <= x < width and 0 <= py_int < height:
                    static_grid[py_int][x] = '='
        
        if gx is not None and gy is not None:
            gx_int = int(gx)
            gy_int = int(gy)
            if 0 <= gx_int < width and 0 <= gy_int < height:
                static_grid[gy_int][gx_int] = 'X'
        
        lines = []
        lines.append("=" * 82)
        lines.append(f" PLATFORMER - SIDE VIEW ({width}x{height}) - Platforms are STATIC!")
        lines.append("=" * 82)
        lines.append(f" Step: {self.step:3d} | Action: {action_name:12s} | Reward: {reward:+.2f} | Total: {self.total_reward:.2f}")
        lines.append(f" Pos: ({px:5.1f},{py:5.1f}) | Vel: ({vel[0]:4.1f},{vel[1]:4.1f}) | Ground: {on_ground} | Facing: {'R' if facing else 'L'}")
        lines.append(f" Goal: {str(goal):15s} | Health: {hud.get('health', 10)}")
        lines.append(f" Result: {str(result)[:60]}")
        lines.append("-" * 82)
        
        # Show viewport around player (side scroller style)
        view_width = 60
        view_height = 25
        min_x = max(0, int(px) - view_width // 2)
        max_x = min(width, min_x + view_width)
        min_y = max(0, int(py) - view_height // 2)
        max_y = min(height, min_y + view_height)
        
        for y in range(min_y, max_y):
            row = ""
            for x in range(min_x, max_x):
                if abs(x - px) < 1 and abs(y - py) < 1:
                    row += "@" if facing else "@"  # Agent
                else:
                    ch = static_grid[y][x] if y < height and x < width else ' '
                    if ch == ' ':
                        row += " "
                    else:
                        row += ch
            lines.append(f" {y:2d} |{row}|")
        
        lines.append("-" * 82)
        lines.append("    " + "".join(str(i % 10) for i in range(min_x, max_x)))
        lines.append("")
        lines.append("Legend: @=Agent  X=Goal  ==Platform  =Empty")
        lines.append("PLATFORMS ARE STATIC - they don't move! Gravity pulls agent down.")
        
        return "\n".join(lines)


async def run_game_demo(game_name, seed=42, max_steps=200, fps=12):
    """Run visual demo for a specific game."""
    
    # Create game
    adapter = create_test_game(game_name, seed=seed)
    
    # Choose visualizer
    if game_name == 'platformer':
        visualizer = PlatformerVisualizer(adapter, game_name)
    else:
        visualizer = GridGameVisualizer(adapter, game_name)
    
    # Create memory with consolidation
    memory = GridMemoryStore(
        game_id=f'visual_{game_name}',
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
    
    # Create exploration policy
    policy = ExplorationPolicy(GridStrategyParameters())
    policy.set_memory(memory)
    
    print("=" * 70)
    print(f"  GAME MASTER - {game_name.upper()} VISUAL DEMO")
    print(f"  Seed: {seed} | Max steps: {max_steps} | FPS: {fps}")
    print("=" * 70)
    print()
    print("Press Ctrl+C to stop")
    print()
    time.sleep(2)
    
    won = False
    
    try:
        while not won and visualizer.step < max_steps:
            obs = adapter.observe()
            
            if obs.get('hud', {}).get('status') != 'playing':
                break
            
            action = policy.decide(obs, memory)
            if action is None:
                available = adapter.get_available_actions()
                if available:
                    action = Action(name=available[0])
                else:
                    break
            
            # Fallback: if action not in available actions, pick first available
            available = adapter.get_available_actions()
            if action.name not in available:
                if available:
                    action = Action(name=available[0])
                else:
                    break
            
            result = adapter.execute(action)
            
            # Check win
            next_obs = adapter.observe()
            won_now = 'won' in str(result.details) or next_obs.get('hud', {}).get('status') == 'won'
            
            # Reward
            if result.success:
                if won_now:
                    reward = 10.0
                elif 'moved' in str(result.details) or 'jump' in str(result.details):
                    reward = 0.2
                else:
                    reward = 0.0
            else:
                reward = -0.05
            
            visualizer.total_reward += reward
            visualizer.step += 1
            
            memory.record_experience({
                'type': 'step',
                'game': game_name,
                'episode': 0,
                'step': visualizer.step,
                'obs': obs,
                'action': action.name,
                'reward': reward,
                'result': str(result.details)
            })
            
            # Render
            clear_screen()
            print(visualizer.render(action.name, result.details, reward))
            
            if memory._consolidator:
                stats = memory._consolidator.get_stats()
                print(f"  STM->MTM: {stats.get('stm_to_mtm', 0)} | MTM->LTM: {stats.get('mtm_to_ltm', 0)} | LTM->GSM: {stats.get('ltm_to_gsm', 0)}")
            
            if won_now:
                print("\n***  AGENT WON!  ***")
                memory.add_knowledge(
                    key=f'win_strategy_{game_name}',
                    value={'final_pos': next_obs.get('hud', {}).get('grid', {}).get('position') or 
                           next_obs.get('hud', {}).get('world', {}).get('position'), 
                           'steps': visualizer.step},
                    confidence=0.9,
                    context={'game': game_name, 'steps': visualizer.step},
                    scope=Scope.GAME,
                    transferable=True
                )
                break
            
            await asyncio.sleep(1.0 / fps)
            
            # Run consolidation every 20 steps
            if visualizer.step % 20 == 0 and memory._consolidator:
                await memory.consolidator.consolidate_step()
                
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    
    print("\n" + "=" * 70)
    print("FINAL STATS")
    print("=" * 70)
    print(f"Steps: {visualizer.step}")
    print(f"Total Reward: {visualizer.total_reward:.2f}")
    print(f"Won: {'YES!' if won else 'No'}")
    
    stats = memory.get_consolidation_stats()
    print(f"\nConsolidation: {stats}")
    
    print("\nKnowledge (GAME scope):")
    for item in memory.get_valid_knowledge(scope=Scope.GAME):
        print(f"  {item.key}: {item.value} (conf={item.confidence:.2f})")


async def main():
    """Main menu to select game."""
    games = [
        ('grid_v1', 'Grid V1 - 30x30 with moving target & pursuer'),
        ('grid_v2', 'Grid V2 - 20x20 with static obstacles & moving goal'),
        ('maze', 'Maze - 21x21 with walls, keys & doors'),
        ('platformer', 'Platformer - 80x30 side-scroller with gravity'),
    ]
    
    print("=" * 70)
    print("  GAME MASTER - VISUAL DEMO SELECTOR")
    print("=" * 70)
    print()
    for i, (name, desc) in enumerate(games, 1):
        print(f"  {i}. {name}: {desc}")
    print("  5. Run ALL games sequentially")
    print()
    
    try:
        choice = input("Select game (1-5): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    
    if choice == '5':
        for name, _ in games:
            await run_game_demo(name, seed=42, max_steps=200, fps=10)
            print("\nPress Enter for next game...")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                break
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(games):
                await run_game_demo(games[idx][0])
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")


if __name__ == '__main__':
    asyncio.run(main())