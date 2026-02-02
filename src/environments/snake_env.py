"""
Snake Game Environment for Deep RL
Implements a Gymnasium-compatible Snake game environment with configurable
state representations and reward shaping.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Any
import copy


class SnakeEnv(gym.Env):
    """
    Snake Game Environment following Gymnasium API.
    
    The agent controls a snake that moves on a grid, eating food to grow.
    Game ends when snake hits walls or itself.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    def __init__(
        self,
        grid_size: int = 20,
        state_representation: str = "feature",
        initial_length: int = 3,
        reward_food: float = 10.0,
        reward_death: float = -10.0,
        reward_step: float = -0.1,
        reward_distance: float = 0.0,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Snake environment.
        
        Args:
            grid_size: Size of the game grid (grid_size x grid_size)
            state_representation: "grid", "feature", or "image"
            initial_length: Initial length of the snake
            reward_food: Reward for eating food
            reward_death: Reward (penalty) for dying
            reward_step: Reward (penalty) per step to encourage efficiency
            reward_distance: Reward scaling for distance to food (0 = disabled)
            render_mode: "human", "rgb_array", or None
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.state_representation = state_representation
        self.initial_length = initial_length
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.reward_distance = reward_distance
        self.render_mode = render_mode
        
        # Action space: 4 discrete actions (UP, DOWN, LEFT, RIGHT)
        self.action_space = spaces.Discrete(4)
        
        # Observation space depends on state representation
        if state_representation == "grid":
            # Full grid as 2D array
            self.observation_space = spaces.Box(
                low=0, high=2, shape=(grid_size, grid_size), dtype=np.int32
            )
        elif state_representation == "feature":
            # Feature vector: [head_x, head_y, food_x, food_y, direction, 
            #                   body_length, danger_straight, danger_left, danger_right]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
            )
        elif state_representation == "image":
            # RGB image representation
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(grid_size, grid_size, 3), dtype=np.uint8
            )
        else:
            raise ValueError(f"Unknown state_representation: {state_representation}")
        
        # Game state
        self.snake = []
        self.food = None
        self.direction = None
        self.score = 0
        self.steps = 0
        self.max_steps = grid_size * grid_size * 2  # Prevent infinite episodes
        
        # Direction vectors: UP, DOWN, LEFT, RIGHT
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize snake at center
        center = self.grid_size // 2
        self.snake = [
            (center, center),
            (center, center - 1),
            (center, center - 2)
        ][:self.initial_length]
        
        # Initial direction: RIGHT
        self.direction = self.RIGHT
        
        # Spawn food
        self._spawn_food()
        
        # Reset counters
        self.score = 0
        self.steps = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps": self.steps
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        
        Returns:
            observation: Next state observation
            reward: Reward for this step
            terminated: Whether episode ended (death)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        # Prevent moving backwards into body
        if action == self.UP and self.direction == self.DOWN:
            action = self.DOWN
        elif action == self.DOWN and self.direction == self.UP:
            action = self.UP
        elif action == self.LEFT and self.direction == self.RIGHT:
            action = self.RIGHT
        elif action == self.RIGHT and self.direction == self.LEFT:
            action = self.LEFT
        
        self.direction = action
        self.steps += 1
        
        # Move snake head
        dx, dy = self.directions[action]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)
        
        # Check for collisions
        terminated = False
        truncated = False
        
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            terminated = True
            reward = self.reward_death
        
        # Self collision
        elif new_head in self.snake[:-1]:  # Exclude tail (will move)
            terminated = True
            reward = self.reward_death
        
        else:
            # Move snake
            self.snake.insert(0, new_head)
            
            # Check if food eaten
            if new_head == self.food:
                self.score += 1
                self._spawn_food()
                reward = self.reward_food
            else:
                # Remove tail if no food eaten
                self.snake.pop()
                reward = self.reward_step
            
            # Add distance-based reward if enabled
            if self.reward_distance > 0 and not terminated:
                old_dist = self._distance_to_food(self.snake[1] if len(self.snake) > 1 else self.snake[0])
                new_dist = self._distance_to_food(new_head)
                distance_reward = (old_dist - new_dist) * self.reward_distance
                reward += distance_reward
            
            # Check max steps
            if self.steps >= self.max_steps:
                truncated = True
        
        # Get next observation
        observation = self._get_observation()
        
        info = {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps": self.steps,
            "food_eaten": new_head == self.food if not terminated else False
        }
        
        return observation, reward, terminated, truncated, info
    
    def _spawn_food(self):
        """Spawn food at random location (not on snake)."""
        while True:
            food = (
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            )
            if food not in self.snake:
                self.food = food
                break
    
    def _distance_to_food(self, pos: Tuple[int, int]) -> float:
        """Calculate Manhattan distance to food."""
        return abs(pos[0] - self.food[0]) + abs(pos[1] - self.food[1])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation based on state representation."""
        if self.state_representation == "grid":
            return self._get_grid_observation()
        elif self.state_representation == "feature":
            return self._get_feature_observation()
        elif self.state_representation == "image":
            return self._get_image_observation()
        else:
            raise ValueError(f"Unknown state_representation: {self.state_representation}")
    
    def _get_grid_observation(self) -> np.ndarray:
        """Get full grid as 2D array (0=empty, 1=snake, 2=food)."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Mark snake body
        for segment in self.snake:
            grid[segment[0], segment[1]] = 1
        
        # Mark food
        if self.food:
            grid[self.food[0], self.food[1]] = 2
        
        return grid
    
    def _get_feature_observation(self) -> np.ndarray:
        """Get feature vector representation."""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Normalize coordinates to [0, 1]
        head_x_norm = head_x / self.grid_size
        head_y_norm = head_y / self.grid_size
        food_x_norm = food_x / self.grid_size
        food_y_norm = food_y / self.grid_size
        
        # Direction as one-hot (4 values)
        direction_onehot = np.zeros(4)
        direction_onehot[self.direction] = 1.0
        
        # Body length (normalized)
        body_length_norm = len(self.snake) / (self.grid_size * self.grid_size)
        
        # Danger detection (straight, left, right relative to current direction)
        danger_straight, danger_left, danger_right = self._check_dangers()
        
        # Combine features
        features = np.array([
            head_x_norm,
            head_y_norm,
            food_x_norm,
            food_y_norm,
            *direction_onehot,
            body_length_norm,
            danger_straight,
            danger_left,
            danger_right
        ], dtype=np.float32)
        
        return features
    
    def _check_dangers(self) -> Tuple[float, float, float]:
        """Check for dangers in straight, left, and right directions."""
        head_x, head_y = self.snake[0]
        dx, dy = self.directions[self.direction]
        
        # Directions relative to current direction
        # Straight: same direction
        # Left: rotate 90 degrees counterclockwise
        # Right: rotate 90 degrees clockwise
        
        def rotate_left(dx, dy):
            return (-dy, dx)
        
        def rotate_right(dx, dy):
            return (dy, -dx)
        
        def check_danger(dx_check, dy_check):
            next_pos = (head_x + dx_check, head_y + dy_check)
            # Check wall
            if (next_pos[0] < 0 or next_pos[0] >= self.grid_size or
                next_pos[1] < 0 or next_pos[1] >= self.grid_size):
                return 1.0
            # Check self collision
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
        """Get RGB image representation."""
        image = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Background: black
        # Snake: green
        for segment in self.snake:
            image[segment[0], segment[1]] = [0, 255, 0]
        
        # Snake head: brighter green
        if self.snake:
            image[self.snake[0][0], self.snake[0][1]] = [0, 255, 100]
        
        # Food: red
        if self.food:
            image[self.food[0], self.food[1]] = [255, 0, 0]
        
        return image
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # For human rendering, we'll use the renderer module
            from .snake_renderer import SnakeRenderer
            if not hasattr(self, '_renderer'):
                self._renderer = SnakeRenderer(self.grid_size)
            self._renderer.render(self.snake, self.food, self.score, self.steps)
        elif self.render_mode == "rgb_array":
            return self._get_image_observation()
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, '_renderer'):
            self._renderer.close()
