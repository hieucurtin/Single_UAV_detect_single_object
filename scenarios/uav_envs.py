from gymnasium import spaces
import numpy as np
from .base_env import BaseUAVEnv

class FullInfoUAVEnv(BaseUAVEnv):
    """
    UAV environment with full object position information.
    Scenario 1: Knows exact object position (x, y coordinates)
    """
    def __init__(self, max_steps=100, render_mode=None):
        super().__init__(max_steps, render_mode)
        # Observation space: UAV position (2D) + object position (2D)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf]),
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
        # Observation space: UAV position (2D) + object direction (2D normalized)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -1.0, -1.0]),
            high=np.array([np.inf, np.inf, 1.0, 1.0]),
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
        # Observation space: UAV position (2D) + detection status (1)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0]),
            high=np.array([np.inf, np.inf, 1]),
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
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),  # Updated to reflect actual bounds
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32
        )

    def _get_obs(self):
        """
        Return noisy observation with Gaussian noise, clipped to [-1, 1]
        
        Returns:
            numpy array: [noisy_uav_x, noisy_uav_y, noisy_object_x, noisy_object_y]
        """
        # Original observation
        obs = np.concatenate([
            self.uav_pos,
            self.object_pos[0]
        ]).astype(np.float32)
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, obs.shape)
        noisy_obs = obs + noise
        # Clip observations to [-1, 1] to match environment bounds
        noisy_obs = np.clip(noisy_obs, -1.0, 1.0)
        return noisy_obs


class NoisyDirectionUAVEnv(BaseUAVEnv):
    """
    Noisy UAV environment with object direction information and different noise levels
    """
    def __init__(self, max_steps=100, render_mode=None, noise_std=0.0):
        super().__init__(max_steps, render_mode)
        self.noise_std = noise_std
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),  # Updated for consistency
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32
        )

    def _get_obs(self):
        """
        Return noisy observation with Gaussian noise on angle, normalized direction
        
        Returns:
            numpy array: [noisy_uav_x, noisy_uav_y, noisy_sin(angle), noisy_cos(angle)]
        """
        angle = np.arctan2(self.object_pos[0][1], self.object_pos[0][0])
        obs = np.concatenate([
            self.uav_pos,
            [np.sin(angle), np.cos(angle)]
        ]).astype(np.float32)
        noise = np.random.normal(0, self.noise_std, obs.shape)
        noisy_obs = obs + noise
        # Normalize direction components to maintain unit vector
        direction = noisy_obs[2:4]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            noisy_obs[2:4] = direction / norm
        # Clip UAV position components to [-1, 1]
        noisy_obs[0:2] = np.clip(noisy_obs[0:2], -1.0, 1.0)
        return noisy_obs