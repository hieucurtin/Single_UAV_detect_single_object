from gymnasium import spaces
import numpy as np
from .base_env import BaseUAVEnv

class FullInfoUAVEnv(BaseUAVEnv):
    """UAV environment with full object position information."""

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
        """Return full state observation."""
        return np.concatenate([
            self.uav_pos,
            self.object_pos[0]
        ]).astype(np.float32)


class DirectionUAVEnv(BaseUAVEnv):
    """UAV environment with object direction information."""

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
        angle = np.arctan2(self.object_pos[0][1], self.object_pos[0][0])
        return np.concatenate([
            self.uav_pos,
            [np.sin(angle), np.cos(angle)]
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
