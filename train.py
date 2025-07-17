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

def evaluate_model(model, env_class, config, scenario_name, algo_name, noise_std=None):
    """
    Evaluate model performance with detailed metrics.
    
    Args:
        model: Trained RL model
        env_class: Environment class to evaluate
        config: Configuration dictionary
        scenario_name: Name of the scenario
        algo_name: Name of the algorithm
        noise_std: Standard deviation of noise for the scenario (optional)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    eval_env = env_class(max_steps=config["max_steps"], render_mode=None, noise_std=noise_std if noise_std is not None else 0.1)
    
    metrics = {
        'scenario': scenario_name,
        'algorithm': algo_name,
        'success_rate': 0,
        'avg_steps_to_detect': 0,
        'avg_reward': 0,
        'std_reward': 0,
        'coverage_efficiency': 0,
        'detection_times': [],
        'rewards': [],
        'step_rewards': [],  # New: Store step-wise rewards for each episode
        'paths': [],
        'object_positions': [],
        'detected_episodes': []
    }
    
    successful_detections = 0
    detection_steps = []
    all_rewards = []
    all_step_rewards = []
    all_paths = []
    all_object_positions = []
    detected_episodes = []
    
    for episode in range(config["eval_episodes"]):
        obs, _ = eval_env.reset(seed=42)
        episode_reward = 0
        episode_steps = 0
        done = False
        path = []
        step_rewards = []
        detected_in_episode = False
        
        while not done and episode_steps < config["max_steps"]:
            action, _ = model.predict(obs,deterministic=False)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            # path.append(obs[:2].copy())
            path.append(eval_env.uav_pos.copy())
            step_rewards.append(reward)
            
            # Check if detection occurred this step
            if info.get("detected_count", 0) > 0 and not detected_in_episode:
                detection_steps.append(episode_steps)
                successful_detections += 1
                detected_in_episode = True
        
        all_rewards.append(episode_reward)
        all_step_rewards.append(step_rewards)
        all_paths.append(path)
        all_object_positions.append([pos.copy() for pos in eval_env.object_pos])
        detected_episodes.append(detected_in_episode)
    
    # Calculate metrics
    metrics['success_rate'] = successful_detections / config["eval_episodes"]
    metrics['avg_reward'] = np.mean(all_rewards)
    metrics['std_reward'] = np.std(all_rewards)
    metrics['avg_steps_to_detect'] = np.mean(detection_steps) if detection_steps else config["max_steps"]
    
    # Calculate coverage efficiency (unique positions visited / total steps)
    total_unique_positions = 0
    total_steps = 0
    for path in all_paths:
        unique_positions = len(set(tuple(pos) for pos in path))
        total_unique_positions += unique_positions
        total_steps += len(path)
    
    metrics['coverage_efficiency'] = total_unique_positions / total_steps if total_steps > 0 else 0
    
    # Store detailed data for visualization
    metrics['detection_times'] = detection_steps
    metrics['rewards'] = all_rewards
    metrics['step_rewards'] = all_step_rewards
    metrics['paths'] = all_paths
    metrics['object_positions'] = all_object_positions
    metrics['detected_episodes'] = detected_episodes
    
    # Initialize UAVVisualization and generate plots
    vis_dir = os.path.join(config["log_dir"], "figures")
    os.makedirs(vis_dir, exist_ok=True)
    visualizer = UAVVisualization(output_dir=vis_dir, noise_std=noise_std)
    visualizer.plot_best_trajectory(all_paths, all_rewards, all_object_positions, all_step_rewards, eval_env)
    visualizer.plot_all_trajectories(all_paths)
    visualizer.plot_detection_stats(detected_episodes)
    visualizer.plot_steps_to_detect(detected_episodes, all_paths, all_step_rewards, eval_env)
    visualizer.plot_policy_visualization(all_rewards, all_object_positions, np.argmax(all_rewards), model, eval_env)
    visualizer.move_figures_to_tensorboard(model)
    
    eval_env.close()
    return metrics

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
    eval_episodes = training_config['eval_episodes']
    
    # Noise levels from config
    noise_levels = training_config['noise_levels']
    
    # Available scenarios from config
    available_scenarios = training_config['scenarios']
    
    # Environment dictionary mapping
    env_dict = {
        'full_info_noise': NoisyFullInfoUAVEnv,
        'direction_noise': NoisyDirectionUAVEnv
    }
    
    # Train and evaluate for each scenario
    for scenario_name in available_scenarios:
        if scenario_name not in env_dict:
            print(f"Warning: Scenario {scenario_name} not found in environment dictionary")
            continue
            
        env_class = env_dict[scenario_name]
        
        # Create environment with noise levels from config
        env = env_class(max_steps=max_steps, render_mode=None,
                       noise_std=noise_levels['noise_medium'])
                       
        # Create and train the model
        model = RLAlgorithmFactory.create_algorithm(
            "PPO",
            env,
            tensorboard_log=tensorboard_log
        )
        
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        # Save the model
        model_path = os.path.join(tensorboard_log, f"{scenario_name}_model")
        model.save(model_path)
        
        # Evaluate the model
        # metrics = evaluate_model(
        #     model,
        #     env_class,
        #     {"max_steps": max_steps, "eval_episodes": eval_episodes, "log_dir": tensorboard_log},
        #     scenario_name,
        #     "PPO",
        #     noise_std=noise_levels['noise_medium']
        # )
        
        # Visualization code here...

if __name__ == "__main__":
    main()