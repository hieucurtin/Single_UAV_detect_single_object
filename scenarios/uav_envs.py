from gymnasium import spaces
import numpy as np
from .base_env import BaseUAVEnv
from utils.config_loader import load_config

class FullInfoUAVEnv(BaseUAVEnv):
    """
    UAV environment with full object position information.
    Scenario 1: Knows exact object position (x, y coordinates)
    """
    def __init__(self, max_steps=100, render_mode=None):
        super().__init__(max_steps, render_mode)
        # Load map bounds from config
        env_config = load_config('env_config')['environment']
        map_bounds_low = np.array(env_config['map_bounds']['low'], dtype=np.float32)
        map_bounds_high = np.array(env_config['map_bounds']['high'], dtype=np.float32)
        # Observation space: UAV position (2D) + object position (2D)
        self.observation_space = spaces.Box(
            low=np.array([map_bounds_low[0], map_bounds_low[1], map_bounds_low[0], map_bounds_low[1]]),
            high=np.array([map_bounds_high[0], map_bounds_high[1], map_bounds_high[0], map_bounds_high[1]]),
            shape=(4,),
            dtype=np.float32
        )

    def _get_obs(self):
        """
        Return full state observation with exact object position.
        
        Returns:
            numpy array: [uav_x, uav_y, object_x, object_y]
        """
        return np.concatenate([
            self.uav_pos,
            self.object_pos[0]
        ]).astype(np.float32)

class DirectionUAVEnv(BaseUAVEnv):
    """
    UAV environment with object direction information.
    Scenario 2: Knows direction to object (angle in coordinate system)
    """
    def __init__(self, max_steps=100, render_mode=None):
        super().__init__(max_steps, render_mode)
        # Load map bounds from config
        env_config = load_config('env_config')['environment']
        map_bounds_low = np.array(env_config['map_bounds']['low'], dtype=np.float32)
        map_bounds_high = np.array(env_config['map_bounds']['high'], dtype=np.float32)
        # Observation space: UAV position (2D) + object direction (2D normalized)
        self.observation_space = spaces.Box(
            low=np.array([map_bounds_low[0], map_bounds_low[1], -1.0, -1.0]),
            high=np.array([map_bounds_high[0], map_bounds_high[1], 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32
        )

    def _get_obs(self):
        """
        Return observation with object direction (angle).
        
        Returns:
            numpy array: [uav_x, uav_y, sin(angle), cos(angle)]
        """
        # Calculate absolute angle of object in coordinate system
        angle = np.arctan2(self.object_pos[0][1], self.object_pos[0][0])
        return np.concatenate([
            self.uav_pos,
            [np.sin(angle), np.cos(angle)]  # Encode angle as unit vector
        ]).astype(np.float32)

class NoInfoUAVEnv(BaseUAVEnv):
    """UAV environment with no object information."""

    def __init__(self, max_steps=100, render_mode=None):
        super().__init__(max_steps, render_mode)
        # Load map bounds from config
        env_config = load_config('env_config')['environment']
        map_bounds_low = np.array(env_config['map_bounds']['low'], dtype=np.float32)
        map_bounds_high = np.array(env_config['map_bounds']['high'], dtype=np.float32)
        # Observation space: UAV position (2D) + detection status (1)
        self.observation_space = spaces.Box(
            low=np.array([map_bounds_low[0], map_bounds_low[1], 0]),
            high=np.array([map_bounds_high[0], map_bounds_high[1], 1]),
            shape=(3,),
            dtype=np.float32
        )

    def _get_obs(self):
        """Return observation with only detection status."""
        return np.concatenate([
            self.uav_pos,
            [float(self.detected["object_0"])]
        ]).astype(np.float32)

class NoisyUAVEnv(BaseUAVEnv):
    """UAV environment with noisy observations."""

    def __init__(self, max_steps=100, render_mode=None, noise_std=0.1, base_env_class=FullInfoUAVEnv):
        super().__init__(max_steps, render_mode)
        self.noise_std = noise_std
        self.base_env_class = base_env_class
        self.base_env = base_env_class(max_steps, render_mode)
        self.observation_space = self.base_env.observation_space

    def _get_obs(self):
        """Return noisy observation."""
        obs = self.base_env._get_obs()
        noise = np.random.normal(0, self.noise_std, obs.shape)
        return (obs + noise).astype(np.float32)

class NoisyFullInfoUAVEnv(BaseUAVEnv):
    """
    Noisy UAV environment with object position information and different noise levels
    """
    def __init__(self, max_steps=100, render_mode=None, noise_std=0.0):
        super().__init__(max_steps, render_mode)
        self.noise_std = noise_std
        # Load map bounds from config
        env_config = load_config('env_config')['environment']
        map_bounds_low = np.array(env_config['map_bounds']['low'], dtype=np.float32)
        map_bounds_high = np.array(env_config['map_bounds']['high'], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([map_bounds_low[0], map_bounds_low[1], map_bounds_low[0], map_bounds_low[1]]),
            high=np.array([map_bounds_high[0], map_bounds_high[1], map_bounds_high[0], map_bounds_high[1]]),
            shape=(4,),
            dtype=np.float32
        )

    def _get_obs(self):
        """
        Return noisy observation with Gaussian noise, clipped to map bounds
        
        Returns:
            numpy array: [noisy_uav_x, noisy_uav_y, noisy_object_x, noisy_object_y]
        """
        # Load map bounds from config
        env_config = load_config('env_config')['environment']
        map_bounds_low = np.array(env_config['map_bounds']['low'], dtype=np.float32)
        map_bounds_high = np.array(env_config['map_bounds']['high'], dtype=np.float32)
        # Original observation
        obs = np.concatenate([
            self.uav_pos,
            self.object_pos[0]
        ]).astype(np.float32)
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, obs.shape)
        noisy_obs = obs + noise
        # Clip observations to map bounds
        noisy_obs = np.clip(noisy_obs, map_bounds_low, map_bounds_high)
        return noisy_obs

class NoisyDirectionUAVEnv(BaseUAVEnv):
    """
    Noisy UAV environment with object direction information and separate noise levels for position and bearing
    """
    def __init__(self, max_steps=100, render_mode=None, position_noise_std=0.0, bearing_noise_std=0.0):
        super().__init__(max_steps, render_mode)
        # Validate noise parameters
        if not isinstance(position_noise_std, (int, float)):
            raise TypeError(f"position_noise_std must be a float or int, got {type(position_noise_std)}: {position_noise_std}")
        if not isinstance(bearing_noise_std, (int, float)):
            raise TypeError(f"bearing_noise_std must be a float or int, got {type(bearing_noise_std)}: {bearing_noise_std}")
        if position_noise_std < 0:
            raise ValueError(f"position_noise_std must be non-negative, got {position_noise_std}")
        if bearing_noise_std < 0:
            raise ValueError(f"bearing_noise_std must be non-negative, got {bearing_noise_std}")
        self.position_noise_std = float(position_noise_std)  # Noise in meters
        self.bearing_noise_std = float(bearing_noise_std) * np.pi / 180.0  # Convert degrees to radians
        # Load map bounds from config
        env_config = load_config('env_config')['environment']
        map_bounds_low = np.array(env_config['map_bounds']['low'], dtype=np.float32)
        map_bounds_high = np.array(env_config['map_bounds']['high'], dtype=np.float32)
        self.map_bounds_low = map_bounds_low
        self.map_bounds_high = map_bounds_high
        self.observation_space = spaces.Box(
            low=np.array([map_bounds_low[0], map_bounds_low[1], -1.0, -1.0]),
            high=np.array([map_bounds_high[0], map_bounds_high[1], 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32
        )

    def _get_obs(self):
        """
        Return noisy observation with Gaussian noise for UAV position (meters) and object direction (degrees converted to radians)
        
        Returns:
            numpy array: [noisy_uav_x, noisy_uav_y, noisy_sin(angle), noisy_cos(angle)]
        """
        # Calculate absolute angle of object in coordinate system
        angle = np.arctan2(self.object_pos[0][1], self.object_pos[0][0])
        # Apply angular noise (in radians)
        noisy_angle = angle + np.random.normal(0, self.bearing_noise_std)
        # Compute observation with true position and noisy direction
        obs = np.concatenate([
            self.uav_pos,
            [np.sin(noisy_angle), np.cos(noisy_angle)]
        ]).astype(np.float32)
        # Apply position noise (in meters)
        noise = np.zeros_like(obs)
        noise[:2] = np.random.normal(0, self.position_noise_std, 2)
        noisy_obs = obs + noise
        # Normalize direction components to maintain unit vector
        direction = noisy_obs[2:4]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            noisy_obs[2:4] = direction / norm
        # Clip UAV position components to map bounds
        noisy_obs[0:2] = np.clip(noisy_obs[0:2], self.map_bounds_low, self.map_bounds_high)
        return noisy_obs