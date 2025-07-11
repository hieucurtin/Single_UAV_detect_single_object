from stable_baselines3 import PPO, SAC
from typing import Dict, Any, Type
from gymnasium import Env

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

        # Default parameters for each algorithm
        default_params = {
            "PPO": {
                "n_steps": 2048,
                "batch_size": 512,
                "ent_coef": 0.05,
                "learning_rate": 5e-4,
                "gamma": 0.99,
            },
            "SAC": {
                "batch_size": 256,
                "ent_coef": 0.1,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
            }
        }

        # Merge default parameters with provided kwargs
        algorithm_params = {**default_params[algo_name], **kwargs}
        algorithm_class = cls.ALGORITHMS[algo_name]

        return algorithm_class(
            "MlpPolicy",
            env,
            tensorboard_log=tensorboard_log,
            **algorithm_params
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
