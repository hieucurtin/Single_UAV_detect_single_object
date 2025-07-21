import os
import argparse
from datetime import datetime
from scenarios.uav_envs import NoisyFullInfoUAVEnv, NoisyDirectionUAVEnv
from utils.config_loader import load_config
from visualization.plotter import UAVVisualization, ComparisonVisualizer
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

def evaluate_trained_model(model, env_class, config, scenario_name, position_noise_std, bearing_noise_std, log_dir):
    """
    Evaluate model performance with detailed metrics.
    
    Args:
        model: Trained RL model
        env_class: Environment class to evaluate
        config: Configuration dictionary
        scenario_name: Name of the scenario
        position_noise_std: Standard deviation of position noise (meters)
        bearing_noise_std: Standard deviation of bearing noise (degrees)
        log_dir: Directory for logging visualizations
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Validate noise parameters
    if not isinstance(position_noise_std, (int, float)):
        raise TypeError(f"position_noise_std must be a float or int, got {type(position_noise_std)}: {position_noise_std}")
    if not isinstance(bearing_noise_std, (int, float)):
        raise TypeError(f"bearing_noise_std must be a float or int, got {type(bearing_noise_std)}: {bearing_noise_std}")
    
    # Initialize the evaluation environment
    eval_env = env_class(max_steps=config["max_steps"], position_noise_std=position_noise_std, bearing_noise_std=bearing_noise_std)
    
    # Lists to store evaluation metrics and trajectories
    all_rewards = []
    all_step_rewards = []
    all_paths = []  # Store true UAV trajectories without noise
    all_object_positions = []
    detected_episodes = []
    detection_steps = []
    successful_detections = 0

    for episode in range(config["eval_episodes"]):
        obs, _ = eval_env.reset()  # Reset with noisy observation
        episode_reward = 0
        episode_steps = 0
        done = False
        path = []  # Store true UAV position for plotting
        step_rewards = []
        detected_in_episode = False
        
        while not done and episode_steps < config["max_steps"]:
            action, _ = model.predict(obs, deterministic=False)  # Predict action using noisy observation
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            obs = next_obs  # Update observation for the next step
            episode_reward += reward
            episode_steps += 1
            path.append(eval_env.uav_pos.copy())  # Use true position without noise for trajectory
            step_rewards.append(reward)
            
            if info.get("detected_count", 0) > 0 and not detected_in_episode:
                detection_steps.append(episode_steps)
                successful_detections += 1
                detected_in_episode = True
        
        all_rewards.append(episode_reward)
        all_step_rewards.append(step_rewards)
        all_paths.append(path)
        all_object_positions.append([pos.copy() for pos in eval_env.object_pos])
        detected_episodes.append(detected_in_episode)

    # Calculate evaluation metrics
    metrics = {
        "scenario": scenario_name,
        "algorithm": "PPO",
        "success_rate": successful_detections / config["eval_episodes"],
        "avg_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "avg_steps_to_detect": np.mean(detection_steps) if detection_steps else config["max_steps"],
        "coverage_efficiency": len(set(tuple(pos) for path in all_paths for pos in path)) / sum(len(path) for path in all_paths) if all_paths else 0,
        "detection_times": detection_steps,
        "rewards": all_rewards,
        "step_rewards": all_step_rewards,
        "paths": all_paths,
        "object_positions": all_object_positions,
        "detected_episodes": detected_episodes
    }
    
    # Visualize results for this scenario
    vis_dir = log_dir
    os.makedirs(vis_dir, exist_ok=True)
    visualizer = UAVVisualization(output_dir=vis_dir, noise_std=(position_noise_std, bearing_noise_std))
    visualizer.plot_best_trajectory(all_paths, all_rewards, all_object_positions, all_step_rewards, eval_env)
    visualizer.plot_all_trajectories(all_paths)
    visualizer.plot_detection_stats(detected_episodes)
    visualizer.plot_steps_to_detect(detected_episodes, all_paths, all_step_rewards, eval_env)
    visualizer.plot_policy_visualization(all_rewards, all_object_positions, np.argmax(all_rewards), model, eval_env)
    
    eval_env.close()
    return metrics

def main():
    # Parse command-line arguments for model directory
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model across noise levels.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the trained model file (e.g., model.zip)")
    args = parser.parse_args()

    # Load configurations
    training_config = load_config('training_config')['training']
    
    # Set up tensorboard logging
    timestamp = datetime.now().strftime("%Y%m%d_%Hhh%Mmm%Sss")
    tensorboard_log = os.path.join(
        training_config['tensorboard']['log_dir'],
        f"eval_{timestamp}"
    )
    
    # Training parameters
    max_steps = training_config['max_steps']
    eval_episodes = training_config['eval_episodes']
    
    # Noise levels from config
    position_noise_levels = training_config['position_noise_levels']
    bearing_noise_levels = training_config['bearing_noise_levels']
    
    # Environment for direction_noise scenario
    env_dict = {
        'full_info_noise': NoisyFullInfoUAVEnv,
        'direction_noise': NoisyDirectionUAVEnv
    }
    scenario_name = 'direction_noise'
    
    # Load the model
    model_path = args.model_dir
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = PPO.load(model_path)
    
    # Dictionary to store results
    results = {scenario_name: {}}
    
    # Evaluate the model across all noise levels
    for noise_level in position_noise_levels.keys():
        position_noise_std = position_noise_levels[noise_level]
        bearing_noise_std = bearing_noise_levels[noise_level]
        print(f"\nEvaluating {scenario_name} with {noise_level} (position std: {position_noise_std}m, bearing std: {bearing_noise_std}°)")
        
        # Evaluate the model
        metrics = evaluate_trained_model(
            model,
            env_dict[scenario_name],
            {"max_steps": max_steps, "eval_episodes": eval_episodes, "log_dir": os.path.join(tensorboard_log, f"{noise_level}")},
            scenario_name=f"{scenario_name}_{noise_level}",
            position_noise_std=position_noise_std,
            bearing_noise_std=bearing_noise_std,
            log_dir=os.path.join(tensorboard_log, f"{noise_level}")
        )
        
        results[scenario_name][noise_level] = metrics
        
        print(f"\nEvaluation Results for {scenario_name} {noise_level} with PPO:")
        print(f"  Success Rate: {metrics['success_rate']:.2f}")
        print(f"  Average Reward: {metrics['avg_reward']:.2f}")
        print(f"  Average Steps to Detect: {metrics['avg_steps_to_detect']:.2f}")
    
    # Generate comparison visualizations
    comparison_dir = os.path.join(tensorboard_log, "comparison_plots")
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_visualizer = ComparisonVisualizer(output_dir=comparison_dir)
    
    if results[scenario_name]:
        comparison_visualizer.generate_all_comparisons(results[scenario_name], scenario_name, single_algo=True)
    else:
        print(f"No results available for {scenario_name} to generate comparisons.")

    print("\nExperiment Results Summary:")
    print(f"\nScenario: {scenario_name}")
    for noise_level, metrics in results[scenario_name].items():
        print(f"  Noise Level: {noise_level}")
        print(f"    Success Rate: {metrics['success_rate']:.2f}")
        print(f"    Average Reward: {metrics['avg_reward']:.2f}")
        print(f"    Average Steps to Detect: {metrics['avg_steps_to_detect']:.2f}")

if __name__ == "__main__":
    main()