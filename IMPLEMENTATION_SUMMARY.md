# Platformer Game - Bug Fixes and Improvements Summary

## Files Modified
- `simulated_game/test_games/platformer_game.py` - Core game implementation
- `tests/test_test_games.py` - Test suite (added 11 new tests)

## Bug Fixes (High Priority)

### B1: Dead Constant `MAX_FALL_SPEED`
**Problem**: `MAX_FALL_SPEED = 12.0` defined but never used; actual limit was `TERMINAL_VELOCITY = 15.0`
**Fix**: Removed `MAX_FALL_SPEED`; set `TERMINAL_VELOCITY = 12.0` as the single max fall speed constant

### B2: Dead Code `_get_platform_at`
**Problem**: Method defined but never called
**Fix**: Removed; collision logic consolidated in `_check_ground_collision()`

### B3: Missing Wall Collision
**Problem**: Player could walk through platform sides (comment admitted "only check ground collision")
**Fix**: Added `_check_wall_collision(old_x, old_y, new_x, vx)` that detects horizontal collision with platform vertical edges and stops player at edge with `vx=0` (wall sliding)

### B4: Jump Doesn't Check Game-Over
**Problem**: `_execute_jump` didn't check `self.won`/`self.caught` after `_step_physics`, unlike `_execute_move`
**Fix**: Added identical post-physics checks for `caught` and `won` in `_execute_jump`

### B5: Stale `on_ground` When Falling
**Problem**: In `_step_physics` falling branch, `on_ground` wasn't updated (remained True)
**Fix**: Explicitly set `on_ground = False` in the `else` branch when not landing on platform

### B6: Inconsistent Spawn/Landing Convention
**Problem**: Spawn at `ground_y - 1`, landing at `ground_y`; ambiguous tolerances
**Fix**: Documented convention in class docstring:
- Player position (x, y) = BOTTOM of player (feet)
- Platform at y=py means surface at py
- When on_ground: player.y == platform_y - 1
- When falling: player.y > platform_y - 1

## Design Improvements (Medium Priority)

### D1: Duplicate Controls
**Problem**: `_CONTROL_BINDINGS` (control_1..3) + direct names in `get_available_actions` → 6 duplicates
**Fix**: Unified to only `control_1` (left), `control_2` (right), `control_3` (jump), `control_4` (wait) in both `observe()["controls"]` and `get_available_actions()`

### D2: Artificially Low Difficulty
**Problem**: `JUMP_FORCE=-8.0`, `GRAVITY=0.5` → max height ≈64, jump distance ≈64 (targets always reachable)
**Fix**: Tuned physics: `GRAVITY=0.8`, `JUMP_FORCE=-5.0` → max height ≈15.6, jump distance ≈15.6

### D3: Jump Not Planable by Planner
**Problem**: BFS planner only emits `move_*`; jump is separate `control_*`
**Fix**: Documented as known limitation; `control_3` (jump) is learnable by ControlLearner

### D4: Incomplete Observation for Decision Making
**Problem**: Missing `target_direction`, `distance_to_goal`, `visible_cells`, `frontier`
**Fix**: Extended `observe()` with:
- `hud.world.target_direction`: [dx, dy] normalized toward goal
- `hud.world.distance_to_goal`: float distance
- `hud.world.visible_cells`: grid cells approximating visible platforms
- `hud.world.frontier`: empty cells adjacent to visible (for exploration)

### D5: Fixed Confidence
**Problem**: `confidence: 1.0` hardcoded
**Fix**: Dynamic: 1.0 if goal visible, 0.5 otherwise

### D6: Unused Health
**Problem**: `hud.health = 10` static, never used
**Fix**: Removed from HUD entirely

### D7: No Reward Shaping
**Problem**: No `distance_progress` for reward learner
**Fix**: Added `distance_progress` (distance before - distance after) to all `ActionResult.details`

## New Features
- Added `control_4` (wait) action with `_execute_wait()` method
- Wait action steps physics (gravity applies) and returns distance_progress

## Tests Added (11 new tests)
1. `test_platformer_physics` - Basic gravity/jump
2. `test_platformer_wall_collision` - B3 wall sliding
3. `test_platformer_fall_off_screen` - B6 fall off = loss
4. `test_platformer_jump_game_over_check` - B4 won/caught check after jump
5. `test_platformer_on_ground_false_when_falling` - B5 on_ground=False when falling
6. `test_platformer_controls_unified` - D1 no duplicate actions
7. `test_platformer_observe_extended` - D4 extended observation fields
8. `test_platformer_confidence_dynamic` - D5 dynamic confidence
9. `test_platformer_distance_progress_reward` - D7 distance_progress in results
10. `test_platformer_playable_to_goal` - Full playthrough < 1000 steps
11. (existing) `test_create_platformer`, `test_platformer_physics`

## Verification
- All 36 tests in `test_test_games.py` pass
- Game playable: simple heuristic reaches goal in ~11 steps (seed 42)
- Physics: gravity, jumping, wall collision, falling all functional
- No regressions in other test games (grid_v1, grid_v2, maze)