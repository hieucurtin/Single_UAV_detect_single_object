from stable_baselines3 import PPO, SAC
from typing import Dict, Any, Type
from gymnasium import Env
from utils.config_loader import load_config

class RLAlgorithmFactory:
    """Factory class for creating RL algorithm instances."""

    ALGORITHMS = {
        "PPO": PPO,
        "SAC": SAC,
    }

    @classmethod
    def create_algorithm(cls, 
                        algo_name: str, 
                        env: Env, 
                        tensorboard_log: str,
                        **kwargs: Dict[str, Any]):
        """
        Create an instance of the specified RL algorithm.
        
        Args:
            algo_name: Name of the algorithm ("PPO" or "SAC")
            env: The training environment
            tensorboard_log: Directory for tensorboard logs
            **kwargs: Additional algorithm-specific parameters
        
        Returns:
            An instance of the specified algorithm
        """
        if algo_name not in cls.ALGORITHMS:
            raise ValueError(f"Algorithm {algo_name} not supported. Available algorithms: {list(cls.ALGORITHMS.keys())}")

        # Load algorithm parameters from config
        agent_config = load_config('agent_config')
        algo_params = agent_config['algorithm_params'][algo_name].copy()
        
        # Override default parameters with any provided kwargs
        algo_params.update(kwargs)
        
        # Add tensorboard log directory
        algo_params['tensorboard_log'] = tensorboard_log

        return cls.ALGORITHMS[algo_name](
            env=env,
            **algo_params
        )

    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type):
        """
        Register a new RL algorithm.
        
        Args:
            name: Name of the algorithm
            algorithm_class: The algorithm class to register
        """
        cls.ALGORITHMS[name] = algorithm_class
