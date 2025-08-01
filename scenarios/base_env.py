import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from utils.config_loader import load_config

class BaseUAVEnv(gym.Env):
    """Base class for UAV object tracking environments."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, max_steps=100, render_mode=None):
        super().__init__()
        # Load environment configuration
        env_config = load_config('env_config')['environment']
        
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.num_objects = env_config['num_objects']
        self.detection_radius = env_config['detection_radius']
        self.tracking_reward = env_config['tracking_reward']
        self.detection_reward = env_config['detection_reward']
        self.step_penalty = env_config['step_penalty']
        self.distance_reward_scale = env_config['distance_reward_scale']
        self.out_bounds_penalty = env_config['out_bounds_penalty']
        self.step_count = 0
        self.detected = {"object_0": False}
        self.episode_reward = 0.0

        # Load map bounds from config
        self.map_bounds_low = np.array(env_config['map_bounds']['low'], dtype=np.float32)
        self.map_bounds_high = np.array(env_config['map_bounds']['high'], dtype=np.float32)

        # UAV and object properties
        self.uav_pos = np.zeros(2, dtype=np.float32)
        self.uav_vel = np.zeros(2, dtype=np.float32)
        self.object_pos = [np.zeros(2, dtype=np.float32)]
        self.max_speed = env_config['max_speed']
        self.damping = env_config['damping']

        # Action space from config
        action_config = env_config['action_space']
        self.action_space = spaces.Box(
            low=np.array(action_config['low'], dtype=np.float32),
            high=np.array(action_config['high'], dtype=np.float32)
        )

        # Rendering setup
        self.screen = None
        self.clock = None
        self.screen_size = 600
        self._render_initialized = False
        self.uav_path = []

    def update_uav_state(self, action):
        """Update UAV position and velocity based on action."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # Update velocity with damping and action contribution
        self.uav_vel = (1 - self.damping) * self.uav_vel + action * self.max_speed
        # Normalize velocity to enforce max_speed
        vel_norm = np.linalg.norm(self.uav_vel)
        if vel_norm > self.max_speed:
            self.uav_vel = self.uav_vel / vel_norm * self.max_speed
        self.uav_pos += self.uav_vel
        self.uav_pos = np.clip(self.uav_pos, self.map_bounds_low, self.map_bounds_high)
        self.uav_path.append(self.uav_pos.copy())

    def compute_reward_cust(self):
        """Compute reward based on UAV and object states."""
        reward = self.step_penalty
        dist = np.linalg.norm(self.uav_pos - self.object_pos[0])
        reward += self.distance_reward_scale * dist

        if dist < self.detection_radius:
            if not self.detected["object_0"]:
                reward += self.detection_reward
                self.detected["object_0"] = True
            reward += self.tracking_reward

        if (abs(self.uav_pos[0]) > (self.map_bounds_high[0]*0.8) or 
            abs(self.uav_pos[1]) > (self.map_bounds_high[1]*0.8)):
            reward += self.out_bounds_penalty

        return float(reward)

    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.detected = {"object_0": False}
        self.uav_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.uav_vel = np.zeros(2, dtype=np.float32)
        self.episode_reward = 0.0
        self.uav_path = [self.uav_pos.copy()]

        self._place_object()
        obs = self._get_obs()
        return obs, {}

    def _place_object(self):
        """Place object in environment. Override in derived classes for different scenarios."""
        angle = np.random.uniform(0, 2 * np.pi)
        # Scale distance to map size, keeping object within 50-80% of map bounds
        distance = np.random.uniform(0.5 * self.map_bounds_high[0], 0.8 * self.map_bounds_high[0])
        self.object_pos[0] = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ], dtype=np.float32)

    def step(self, action):
        """Execute one time step in the environment."""
        self.step_count += 1
        self.update_uav_state(action)

        reward = self.compute_reward_cust()
        self.episode_reward += reward

        obs = self._get_obs()
        terminated = False
        truncated = self.step_count >= self.max_steps or self.detected["object_0"]
        info = {
            "detected_count": sum(self.detected.values()),
            "uav_path": self.uav_path.copy()
        }

        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.step_count,
            }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Get observation. Must be implemented by derived classes."""
        raise NotImplementedError

    def render(self, uav_path=None):
        """Render environment state."""
        if self.render_mode is None:
            return

        if not self._render_initialized:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            else:  # "rgb_array"
                self.screen = pygame.Surface((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()
            self._render_initialized = True

        self.screen.fill((255, 255, 255))

        # Convert coordinates from map_bounds to screen coordinates
        def to_screen(pos):
            # Normalize position to [0, 1] based on map bounds
            norm_x = (pos[0] - self.map_bounds_low[0]) / (self.map_bounds_high[0] - self.map_bounds_low[0])
            norm_y = (pos[1] - self.map_bounds_low[1]) / (self.map_bounds_high[1] - self.map_bounds_low[1])
            return (
                int(norm_x * self.screen_size),
                int(norm_y * self.screen_size)
            )

        # Draw detection radius
        center = to_screen(self.uav_pos)
        # Scale radius based on map size
        radius = int(self.detection_radius * self.screen_size / (self.map_bounds_high[0] - self.map_bounds_low[0]))
        pygame.draw.circle(self.screen, (200, 200, 200), center, radius)

        # Draw UAV path
        if uav_path:
            path_points = [to_screen(pos) for pos in uav_path]
            if len(path_points) > 1:
                pygame.draw.lines(self.screen, (0, 0, 255), False, path_points)

        # Draw object
        obj_pos = to_screen(self.object_pos[0])
        pygame.draw.circle(self.screen, (255, 0, 0), obj_pos, 10)

        # Draw UAV
        pygame.draw.circle(self.screen, (0, 255, 0), center, 10)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)),
            axes=(1, 0, 2)
        ) if self.render_mode == "rgb_array" else None

    def close(self):
        """Clean up resources."""
        if self._render_initialized:
            if self.screen is not None:
                pygame.display.quit()
            pygame.quit()
            self._render_initialized = False
