import os
from scenarios.uav_envs import FullInfoUAVEnv, DirectionUAVEnv, NoInfoUAVEnv, NoisyUAVEnv
from scenarios.uav_envs import NoisyFullInfoUAVEnv, NoisyDirectionUAVEnv
from utils.rl_factory import RLAlgorithmFactory
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
    # Configuration
    config = {
        "max_steps": 100,
        "total_timesteps": 100000,
        "eval_episodes": 100,
        "algorithm_params": {
            "PPO": {
                "n_steps": 2048,
                "batch_size": 512,
                "ent_coef": 0.01,
                "gamma" : 0.99,
                "ent_coef" : 0.01,
                "learning_rate" : 0.0005
            },
            "SAC": {
                "batch_size": 256,
                "ent_coef": "auto",
            }
        }
    }

    full_info_noise_scenarios = {
        "noise_low": {"env_factory": lambda **kwargs: NoisyFullInfoUAVEnv(**kwargs), "noise_std": 0},  # Reduced from 0.075
        "noise_medium": {"env_factory": lambda **kwargs: NoisyFullInfoUAVEnv(**kwargs), "noise_std": 0},  # Reduced from 0.35
        "noise_high": {"env_factory": lambda **kwargs: NoisyFullInfoUAVEnv(**kwargs), "noise_std": 0}  # Reduced from 1.5
    }
    direction_noise_scenarios = {
        "noise_low": {"env_factory": lambda **kwargs: NoisyDirectionUAVEnv(**kwargs), "noise_std": 0.0},  # Reduced from 0.025
        "noise_medium": {"env_factory": lambda **kwargs: NoisyDirectionUAVEnv(**kwargs), "noise_std": 0.0},  # Reduced from 0.15
        "noise_high": {"env_factory": lambda **kwargs: NoisyDirectionUAVEnv(**kwargs), "noise_std": 0.0}  # Reduced from 0.4
    }

    # Available algorithms
    algorithms = ["PPO"]

    # Dictionary to store results
    results = {}

    # Create a unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%Hhh%Mmm%Sss")
    base_log_dir = os.path.join("_tensorboard_logs", f"{timestamp}")
    os.makedirs(base_log_dir, exist_ok=True)

    # Visualization comparison
    comparison_visualizer = ComparisonVisualizer(
        output_dir=os.path.join(base_log_dir, "comparison_plots")
    )

    # Run experiments for noise scenarios of full info environment
    results["full_info_noise"] = {}
    for scenario_name, scenario_info in full_info_noise_scenarios.items():
        results["full_info_noise"][scenario_name] = {}
        env_factory = scenario_info["env_factory"]
        noise_std = scenario_info["noise_std"]
        
        for algo_name in algorithms:
            print(f"\nRunning Full Info {scenario_name} scenario with {algo_name}")
            
            # Set a unique log directory
            log_dir = os.path.join(base_log_dir, f"{algo_name}_logs", "full_info_noise", scenario_name)
            config["log_dir"] = log_dir
            os.makedirs(log_dir, exist_ok=True)
            # Train the model
            try:
                model = RLAlgorithmFactory.create_algorithm(algo_name, env_factory(max_steps=config["max_steps"], noise_std=noise_std), tensorboard_log=log_dir, **config.get("algorithm_params", {}).get(algo_name, {}))
                model.learn(total_timesteps=config["total_timesteps"], progress_bar=True)
                # metrics = evaluate_model(model, env_factory, config, scenario_name, algo_name, noise_std=noise_std)
                # results["full_info_noise"][scenario_name][algo_name] = metrics
                model_path = os.path.join(log_dir, f"full_info_{scenario_name}_{algo_name}_model")
                model.save(model_path)
                # comparison_visualizer.move_figures_to_tensorboard(model)
                # print(f"Saved metrics for {algo_name} in {scenario_name}: success_rate={metrics['success_rate']:.2f}")
            except Exception as e:
                print(f"Error running {algo_name} for {scenario_name}: {e}")
                raise

    # Run experiments for noise scenarios of direction environment
    results["direction_noise"] = {}
    for scenario_name, scenario_info in direction_noise_scenarios.items():
        results["direction_noise"][scenario_name] = {}
        env_factory = scenario_info["env_factory"]
        noise_std = scenario_info["noise_std"]
        
        for algo_name in algorithms:
            print(f"\nRunning Direction {scenario_name} scenario with {algo_name}")
            
            # Set a unique log directory
            log_dir = os.path.join(base_log_dir, f"{algo_name}_logs", "direction_noise", scenario_name)
            config["log_dir"] = log_dir
            os.makedirs(log_dir, exist_ok=True)

            try:
                model = RLAlgorithmFactory.create_algorithm(algo_name, env_factory(max_steps=config["max_steps"], noise_std=noise_std), tensorboard_log=log_dir, **config.get("algorithm_params", {}).get(algo_name, {}))
                model.learn(total_timesteps=config["total_timesteps"], progress_bar=True)
                # metrics = evaluate_model(model, env_factory, config, scenario_name, algo_name, noise_std=noise_std)
                # results["direction_noise"][scenario_name][algo_name] = metrics
                model_path = os.path.join(log_dir, f"direction_{scenario_name}_{algo_name}_model")
                model.save(model_path)
                # comparison_visualizer.move_figures_to_tensorboard(model)
                # print(f"Saved metrics for {algo_name} in {scenario_name}: success_rate={metrics['success_rate']:.2f}")
            except Exception as e:
                print(f"Error running {algo_name} for {scenario_name}: {e}")
                raise

    print(f"\nResults for full_info_noise: {results['full_info_noise']}")
    if not results["full_info_noise"]:
        raise ValueError("No results for full_info_noise. Check training and evaluation steps.")
    comparison_visualizer.generate_all_comparisons(results["full_info_noise"], "full_info_noise")

    # print(f"Results for direction_noise: {results['direction_noise']}")
    if not results["direction_noise"]:
        raise ValueError("No results for direction_noise. Check training and evaluation steps.")
    comparison_visualizer.generate_all_comparisons(results["direction_noise"], "direction_noise")

    print("\nExperiment Results Summary:")
    for noise_category, scenarios in results.items():
        print(f"\nNoise Category: {noise_category}")
        for scenario, algo_results in scenarios.items():
            print(f"  Scenario: {scenario}")
            for algo, metrics in algo_results.items():
                print(f"    Algorithm: {algo}")
                print(f"      Success Rate: {metrics['success_rate']:.2f}")
                print(f"      Average Reward: {metrics['avg_reward']:.2f}")
                print(f"      Average Steps to Detect: {metrics['avg_steps_to_detect']:.2f}")

    return results

if __name__ == "__main__":
    from datetime import datetime
    results = main()