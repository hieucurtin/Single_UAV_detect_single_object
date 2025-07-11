import os
from scenarios.uav_envs import FullInfoUAVEnv, DirectionUAVEnv, NoInfoUAVEnv, NoisyUAVEnv
from utils.rl_factory import RLAlgorithmFactory
from visualization.plotter import UAVVisualization
from visualization.plotter import ComparisonVisualizer
from stable_baselines3.common.env_checker import check_env
import numpy as np
import matplotlib.pyplot as plt

def train_and_evaluate(env_class, algo_name, config, log_dir):
    """Train and evaluate a specific scenario with a specific algorithm."""
    # Create and validate environment
    env = env_class(max_steps=config["max_steps"], render_mode=None)
    check_env(env)

    # Create and train the model
    print(f"Starting training with {algo_name}...")
    model = RLAlgorithmFactory.create_algorithm(
        algo_name,
        env,
        tensorboard_log=log_dir,
        **config.get("algorithm_params", {}).get(algo_name, {})
    )
    
    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True
    )
    
    # Evaluation
    print("\nStarting evaluation...")
    eval_env = env_class(max_steps=config["max_steps"])
    
    paths = []
    rewards = []
    detected_list = []
    rewards_list = []
    object_positions_list = []
    
    for episode in range(config["eval_episodes"]):
        obs, _ = eval_env.reset(seed=episode)
        episode_reward = 0
        done = False
        path = []
        episode_rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            path.append(obs[:2])  # Store UAV position
            episode_rewards.append(reward)
            
        paths.append(path)
        rewards.append(episode_reward)
        detected_list.append(info["detected_count"] > 0)
        rewards_list.append(episode_rewards)
        object_positions_list.append([pos.copy() for pos in eval_env.object_pos])
    
    # Create visualizer and generate plots
    visualizer = UAVVisualization()
    
    # Plot all visualizations
    visualizer.plot_best_trajectory(paths, rewards, object_positions_list, rewards_list, eval_env)
    visualizer.plot_all_trajectories(paths)
    visualizer.plot_detection_stats(detected_list)
    visualizer.plot_steps_to_detect(detected_list, paths, rewards_list, eval_env)
    # visualizer.plot_policy_visualization(rewards, object_positions_list, np.argmax(rewards), model, eval_env)
    
    # Move figures to tensorboard directory
    visualizer.move_figures_to_tensorboard(model)
    
    eval_env.close()
    return paths, rewards, detected_list

def evaluate_model(model, env_class, config, scenario_name, algo_name):
    """
    Evaluate model performance with detailed metrics.
    
    Args:
        model: Trained RL model
        env_class: Environment class to evaluate
        config: Configuration dictionary
        scenario_name: Name of the scenario
        algo_name: Name of the algorithm
    
    Returns:
        Dictionary containing evaluation metrics
    """
    eval_env = env_class(max_steps=config["max_steps"])
    
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
        'paths': [],
        'object_positions': [],
        'detected_episodes': []
    }
    
    successful_detections = 0
    detection_steps = []
    all_rewards = []
    all_paths = []
    all_object_positions = []
    detected_episodes = []
    
    for episode in range(config["eval_episodes"]):
        obs, _ = eval_env.reset(seed=episode)
        episode_reward = 0
        episode_steps = 0
        done = False
        path = []
        detected_in_episode = False
        
        while not done and episode_steps < config["max_steps"]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            path.append(obs[:2].copy())  # Store UAV position
            
            # Check if detection occurred this step
            if info.get("detected_count", 0) > 0 and not detected_in_episode:
                detection_steps.append(episode_steps)
                successful_detections += 1
                detected_in_episode = True
        
        all_rewards.append(episode_reward)
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
    metrics['paths'] = all_paths
    metrics['object_positions'] = all_object_positions
    metrics['detected_episodes'] = detected_episodes
    
    eval_env.close()
    return metrics

def main():
    # Configuration
    config = {
        "max_steps": 100,
        "total_timesteps": 1000,
        "eval_episodes": 100,
        "algorithm_params": {
            "PPO": {
                "n_steps": 1024,
                "batch_size": 256,
                "ent_coef": 0.01,
            },
            "SAC": {
                "batch_size": 256,
                "ent_coef": "auto",
            }
        }
    }

    # Create a unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join("_tensorboard_logs", f"run_{timestamp}")
    os.makedirs(base_log_dir, exist_ok=True)

    # Available scenarios
    scenarios = {
        "full_info": FullInfoUAVEnv,
        "direction_only": DirectionUAVEnv,
        "no_info": NoInfoUAVEnv,
        "noisy": lambda **kwargs: NoisyUAVEnv(base_env_class=FullInfoUAVEnv, noise_std=0.1, **kwargs)
    }

    # Available algorithms
    # algorithms = ["PPO", "SAC"]
    algorithms = ["PPO"]

    # Dictionary to store results
    results = {}

    # Visualization comparison
    comparison_visualizer = ComparisonVisualizer(
        output_dir=os.path.join(base_log_dir, "comparison_plots")
    )

    # Run experiments for each scenario and algorithm
    for scenario_name, env_class in scenarios.items():
        results[scenario_name] = {}
        
        for algo_name in algorithms:
            print(f"\nRunning {scenario_name} scenario with {algo_name}")
            
            # Set a unique log directory for this scenario and algorithm
            log_dir = os.path.join(base_log_dir, f"{algo_name}_logs", scenario_name)
            os.makedirs(log_dir, exist_ok=True)

            # Train the model
            model = RLAlgorithmFactory.create_algorithm(
                algo_name,
                env_class(max_steps=config["max_steps"]),
                tensorboard_log=log_dir,
                **config.get("algorithm_params", {}).get(algo_name, {})
            )

            # Learn
            model.learn(
                total_timesteps=config["total_timesteps"],
                progress_bar=True
            )

            # Evaluate the model
            metrics = evaluate_model(
                model, 
                env_class, 
                config, 
                scenario_name, 
                algo_name
            )

            # Store results
            results[scenario_name][algo_name] = metrics

            # Save the model
            model_path = os.path.join(log_dir, f"{scenario_name}_{algo_name}_model")
            model.save(model_path)

            # Move figures to the TensorBoard log directory
            comparison_visualizer.move_figures_to_tensorboard(model)

    # Generate comprehensive visualizations
    comparison_visualizer.generate_all_comparisons(results)

    # Optional: Print a summary of results
    print("\nExperiment Results Summary:")
    for scenario, algo_results in results.items():
        print(f"\nScenario: {scenario}")
        for algo, metrics in algo_results.items():
            print(f"  Algorithm: {algo}")
            print(f"    Success Rate: {metrics['success_rate']:.2f}")
            print(f"    Average Reward: {metrics['avg_reward']:.2f}")
            print(f"    Average Steps to Detect: {metrics['avg_steps_to_detect']:.2f}")

    return results

if __name__ == "__main__":
    # Import datetime for timestamp
    from datetime import datetime
    
    results = main()

