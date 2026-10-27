from collections import deque

from core.types import Action, GameState


class Planner:
    def choose_action(self, state: GameState) -> Action | None:
        if state is None:
            return None

        objects = state.objects or []
        objective = state.goal

        if objective == "reach_goal":
            action = self._choose_grid_action(state.hud.get("grid"))
            if action is not None:
                return action

        if "start_button" in objects:
            return Action(name="click", params={"x": 500, "y": 400})

        if objective == "move_left":
            return Action(name="move_left", params={"steps": 1})

        if objective == "observe":
            return Action(name="wait", params={"duration": 0.1})

        if "unknown" in objects:
            return Action(name="wait", params={"duration": 0.1})

        return Action(name="wait", params={"duration": 0.1})

    @staticmethod
    def _choose_grid_action(grid: object) -> Action | None:
        if not isinstance(grid, dict):
            return None

        width = grid.get("width")
        height = grid.get("height")
        start = grid.get("position")
        target = grid.get("target")
        obstacles = grid.get("obstacles")
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or width < 1
            or height < 1
            or not isinstance(start, list)
            or not isinstance(target, list)
            or len(start) != 2
            or len(target) != 2
            or not all(isinstance(value, int) for value in start + target)
            or not isinstance(obstacles, list)
        ):
            return None

        start_position = (start[0], start[1])
        target_position = (target[0], target[1])
        blocked: set[tuple[int, int]] = set()
        for obstacle in obstacles:
            if (
                not isinstance(obstacle, list)
                or len(obstacle) != 2
                or not all(isinstance(value, int) for value in obstacle)
            ):
                return None
            blocked.add((obstacle[0], obstacle[1]))

        if (
            not (0 <= start_position[0] < width and 0 <= start_position[1] < height)
            or not (0 <= target_position[0] < width and 0 <= target_position[1] < height)
            or start_position in blocked
            or target_position in blocked
        ):
            return None
        if start_position == target_position:
            return Action(name="wait", params={"duration": 0.1})

        moves = (
            (0, -1, "move_up"),
            (1, 0, "move_right"),
            (0, 1, "move_down"),
            (-1, 0, "move_left"),
        )
        queue = deque([(start_position, None)])
        visited = {start_position}

        while queue:
            (x, y), first_action = queue.popleft()
            for dx, dy, action_name in moves:
                next_position = (x + dx, y + dy)
                if (
                    not (0 <= next_position[0] < width and 0 <= next_position[1] < height)
                    or next_position in blocked
                    or next_position in visited
                ):
                    continue
                next_action = first_action or action_name
                if next_position == target_position:
                    return Action(name=next_action)
                visited.add(next_position)
                queue.append((next_position, next_action))

        return None
