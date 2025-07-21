import os
from datetime import datetime
from scenarios.uav_envs import FullInfoUAVEnv, DirectionUAVEnv, NoInfoUAVEnv, NoisyUAVEnv
from scenarios.uav_envs import NoisyFullInfoUAVEnv, NoisyDirectionUAVEnv
from utils.rl_factory import RLAlgorithmFactory
from utils.config_loader import load_config
from visualization.plotter import UAVVisualization
from visualization.plotter import ComparisonVisualizer
from stable_baselines3.common.env_checker import check_env
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load configurations
    training_config = load_config('training_config')['training']
    
    # Set up tensorboard logging
    timestamp = datetime.now().strftime("%Y%m%d_%Hhh%Mmm%Sss")
    tensorboard_log = os.path.join(
        training_config['tensorboard']['log_dir'],
        training_config['tensorboard']['run_name'] or timestamp
    )
    
    # Training parameters
    max_steps = training_config['max_steps']
    total_timesteps = training_config['total_timesteps']
    
    # Noise levels from config
    position_noise_levels = training_config['position_noise_levels']
    bearing_noise_levels = training_config['bearing_noise_levels']
    
    # Environment dictionary mapping
    env_dict = {
        'full_info_noise': NoisyFullInfoUAVEnv,
        'direction_noise': NoisyDirectionUAVEnv
    }
    
    # Train for direction noise scenario with different noise levels
    scenario_name = 'direction_noise'
    env_class = env_dict[scenario_name]
    
    for noise_level in position_noise_levels.keys():
        position_noise_std = position_noise_levels[noise_level]
        bearing_noise_std = bearing_noise_levels[noise_level]
        print(f"\nTraining {scenario_name} with {noise_level} (position std: {position_noise_std}m, bearing std: {bearing_noise_std}°)")
        
        # Validate noise parameters
        if not isinstance(position_noise_std, (int, float)):
            raise TypeError(f"position_noise_std must be a float or int, got {type(position_noise_std)}: {position_noise_std}")
        if not isinstance(bearing_noise_std, (int, float)):
            raise TypeError(f"bearing_noise_std must be a float or int, got {type(bearing_noise_std)}: {bearing_noise_std}")
        
        # Create environment with current noise levels
        env = env_class(max_steps=max_steps, render_mode=None,
                        position_noise_std=position_noise_std,
                        bearing_noise_std=bearing_noise_std)
                       
        # Create and train the model
        model = RLAlgorithmFactory.create_algorithm(
            "PPO",
            env,
            tensorboard_log=os.path.join(tensorboard_log, f"{noise_level}")
        )
        
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        # Create a directory for the model and configs
        save_dir = os.path.join(tensorboard_log, f"{scenario_name}_{noise_level}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(save_dir, "model")
        model.save(model_path)
        
        # Copy configuration files
        import shutil
        config_dir = os.path.join(save_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        
        # Copy all config files
        config_files = ['agent_config.yaml', 'env_config.yaml', 'training_config.yaml']
        for config_file in config_files:
            src = os.path.join('config', config_file)
            dst = os.path.join(config_dir, config_file)
            shutil.copy2(src, dst)
            
        print(f"Saved model and configurations to {save_dir}")

if __name__ == "__main__":
    main()