"""
Snake Game Environment for Deep RL
Gymnasium-compatible: move, eat, grow, die on wall or self-collision.

Core rewards: food / death / step. Optional reward_distance scales (old_dist - new_dist) so moving
toward food is rewarded (reduces circling).
"""

import inspect
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Any


class SnakeEnv(gym.Env):
    """
    Snake on a grid: actions are UP/DOWN/LEFT/RIGHT. Opposite of current motion is ignored
    unless that cell has food (no instant 180° into body).

    Rewards: death, food, per-step; plus optional reward_distance * (Manhattan improvement toward food)
    on non-terminal steps that do not eat (eating uses reward_food only so new food does not confuse shaping).

    Feature observations use food projected onto forward / left axes (ego-centric), not world dx/dy, so
    actions align with "toward food" in the snake's frame.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(
        self,
        grid_size: int = 20,
        state_representation: str = "feature",
        initial_length: int = 3,
        reward_food: float = 1.0,
        reward_death: float = -1.0,
        reward_step: float = 0.0,
        reward_distance: float = 0.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.state_representation = state_representation
        self.initial_length = initial_length
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.reward_distance = reward_distance
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)

        if state_representation == "grid":
            self.observation_space = spaces.Box(
                low=0, high=2, shape=(grid_size, grid_size), dtype=np.int32
            )
        elif state_representation == "feature":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
            )
        elif state_representation == "image":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(grid_size, grid_size, 3), dtype=np.uint8
            )
        else:
            raise ValueError(f"Unknown state_representation: {state_representation}")

        self.snake = []
        self.food = None
        self.direction = None
        self.score = 0
        self.steps = 0
        self.max_steps = grid_size * grid_size * 2

        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    @classmethod
    def from_config(
        cls,
        env_cfg: Dict[str, Any],
        render_mode: Optional[str] = None,
    ) -> "SnakeEnv":
        """Build env from a config ``environment`` dict (delegates to :func:`snake_env_from_config`)."""
        return snake_env_from_config(env_cfg, render_mode=render_mode)

    @staticmethod
    def _cells_equal(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return int(a[0]) == int(b[0]) and int(a[1]) == int(b[1])

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def _is_opposite_direction(self, action: int, current_dir: int) -> bool:
        return (
            (action == self.UP and current_dir == self.DOWN)
            or (action == self.DOWN and current_dir == self.UP)
            or (action == self.LEFT and current_dir == self.RIGHT)
            or (action == self.RIGHT and current_dir == self.LEFT)
        )

    def _resolve_action(self, raw_action: int) -> int:
        """No 180° turn into body, unless the target cell is food."""
        action = int(raw_action)
        head_x, head_y = self.snake[0]
        if self._is_opposite_direction(action, self.direction):
            dx_t, dy_t = self.directions[action]
            tentative = (head_x + dx_t, head_y + dy_t)
            if self.food is None or not self._cells_equal(tentative, self.food):
                action = self.direction
        return action

    @staticmethod
    def _in_bounds(pos: Tuple[int, int], grid_size: int) -> bool:
        return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size

    def _try_move(
        self, head_x: int, head_y: int, raw_action: int
    ) -> Optional[Tuple[Tuple[int, int], int]]:
        """Valid move: (new_head, final_direction), or None if wall or body."""
        a = self._resolve_action(int(raw_action))
        dx, dy = self.directions[a]
        nh = (head_x + dx, head_y + dy)
        if not self._in_bounds(nh, self.grid_size):
            return None
        if nh in self.snake[:-1]:
            return None
        return (nh, a)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        center = self.grid_size // 2
        self.snake = [
            (center, center),
            (center, center - 1),
            (center, center - 2),
        ][: self.initial_length]

        self.direction = self.RIGHT
        self._spawn_food()

        self.score = 0
        self.steps = 0

        observation = self._get_observation()
        info = {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps": self.steps,
        }
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps += 1

        head_x, head_y = self.snake[0]
        food_before = self.food
        old_dist = (
            self._manhattan((head_x, head_y), food_before)
            if food_before is not None
            else 0
        )

        move = self._try_move(head_x, head_y, int(action))

        terminated = False
        truncated = False
        food_eaten = False

        if move is None:
            terminated = True
            reward = self.reward_death
        else:
            new_head, final_direction = move
            self.direction = final_direction
            self.snake.insert(0, new_head)

            food_eaten = self._cells_equal(new_head, self.food)
            if food_eaten:
                self.score += 1
                self._spawn_food()
                reward = self.reward_food
            else:
                self.snake.pop()
                new_dist = (
                    self._manhattan(new_head, self.food)
                    if self.food is not None
                    else 0
                )
                reward = self.reward_step + self.reward_distance * (old_dist - new_dist)

            if self.steps >= self.max_steps:
                truncated = True

        observation = self._get_observation()
        info = {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps": self.steps,
            "food_eaten": food_eaten if not terminated else False,
        }
        return observation, reward, terminated, truncated, info

    def _spawn_food(self) -> None:
        while True:
            food = (
                int(self.np_random.integers(0, self.grid_size)),
                int(self.np_random.integers(0, self.grid_size)),
            )
            if food not in self.snake:
                self.food = food
                break

    def _get_observation(self) -> np.ndarray:
        if self.state_representation == "grid":
            return self._get_grid_observation()
        if self.state_representation == "feature":
            return self._get_feature_observation()
        if self.state_representation == "image":
            return self._get_image_observation()
        raise ValueError(f"Unknown state_representation: {self.state_representation}")

    def _get_grid_observation(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for segment in self.snake:
            grid[segment[0], segment[1]] = 1
        if self.food:
            grid[self.food[0], self.food[1]] = 2
        return grid

    def _food_ego_forward_left(self) -> Tuple[float, float]:
        """Food offset projected onto forward and left axes (same frame as actions / dangers)."""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dr = float(food_x - head_x)
        dc = float(food_y - head_y)
        fdx, fdy = self.directions[self.direction]
        ldx, ldy = -fdy, fdx
        inv = float(self.grid_size)
        forward = (dr * fdx + dc * fdy) / inv
        left = (dr * ldx + dc * ldy) / inv
        return forward, left

    def _get_feature_observation(self) -> np.ndarray:
        head_x, head_y = self.snake[0]

        head_x_norm = head_x / self.grid_size
        head_y_norm = head_y / self.grid_size
        food_forward_norm, food_left_norm = self._food_ego_forward_left()

        direction_onehot = np.zeros(4)
        direction_onehot[self.direction] = 1.0

        body_length_norm = len(self.snake) / (self.grid_size * self.grid_size)

        danger_straight, danger_left, danger_right = self._check_dangers()

        return np.array(
            [
                head_x_norm,
                head_y_norm,
                food_forward_norm,
                food_left_norm,
                *direction_onehot,
                body_length_norm,
                danger_straight,
                danger_left,
                danger_right,
            ],
            dtype=np.float32,
        )

    def _check_dangers(self) -> Tuple[float, float, float]:
        head_x, head_y = self.snake[0]
        dx, dy = self.directions[self.direction]

        def rotate_left(ddx, ddy):
            return (-ddy, ddx)

        def rotate_right(ddx, ddy):
            return (ddy, -ddx)

        def check_danger(dx_check, dy_check):
            next_pos = (head_x + dx_check, head_y + dy_check)
            if (
                next_pos[0] < 0
                or next_pos[0] >= self.grid_size
                or next_pos[1] < 0
                or next_pos[1] >= self.grid_size
            ):
                return 1.0
            if next_pos in self.snake[:-1]:
                return 1.0
            return 0.0

        danger_straight = check_danger(dx, dy)
        left_dx, left_dy = rotate_left(dx, dy)
        danger_left = check_danger(left_dx, left_dy)
        right_dx, right_dy = rotate_right(dx, dy)
        danger_right = check_danger(right_dx, right_dy)
        return danger_straight, danger_left, danger_right

    def _get_image_observation(self) -> np.ndarray:
        image = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for segment in self.snake:
            image[segment[0], segment[1]] = [0, 255, 0]
        if self.snake:
            image[self.snake[0][0], self.snake[0][1]] = [0, 255, 100]
        if self.food:
            image[self.food[0], self.food[1]] = [255, 0, 0]
        return image

    def render(self):
        if self.render_mode == "human":
            from .snake_renderer import SnakeRenderer

            if getattr(self, "_renderer", None) is None:
                self._renderer = SnakeRenderer(self.grid_size)
            self._renderer.render(self.snake, self.food, self.score, self.steps)
        elif self.render_mode == "rgb_array":
            return self._get_image_observation()

    def close(self):
        r = getattr(self, "_renderer", None)
        if r is not None:
            r.close()
            self._renderer = None


def snake_env_from_config(
    env_cfg: Dict[str, Any],
    render_mode: Optional[str] = None,
) -> SnakeEnv:
    """
    Build a :class:`SnakeEnv` from a config ``environment`` dict.
    Ignores keys the constructor does not accept (safe across versions).
    Prefer this in notebooks after ``importlib.reload(environments.snake_env)`` if the kernel cached an old class.
    """
    kwargs = {
        "grid_size": env_cfg["grid_size"],
        "state_representation": env_cfg["state_representation"],
        "initial_length": env_cfg["initial_length"],
        "reward_food": env_cfg["reward_food"],
        "reward_death": env_cfg["reward_death"],
        "reward_step": env_cfg["reward_step"],
        "reward_distance": env_cfg.get("reward_distance", 0.0),
    }
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    allowed = set(inspect.signature(SnakeEnv.__init__).parameters) - {"self"}
    return SnakeEnv(**{k: v for k, v in kwargs.items() if k in allowed})
