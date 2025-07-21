import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
from scenarios.uav_envs import NoisyFullInfoUAVEnv, NoisyDirectionUAVEnv
from utils.config_loader import load_config

class UAVVisualization:
    def __init__(self, output_dir="_plots", noise_std=None):
        self.output_dir = output_dir
        self.noise_std = noise_std  # Tuple of (position_noise_std, bearing_noise_std) or single noise_std
        # Load map bounds from config
        env_config = load_config('env_config')['environment']
        self.map_bounds_low = np.array(env_config['map_bounds']['low'], dtype=np.float32)
        self.map_bounds_high = np.array(env_config['map_bounds']['high'], dtype=np.float32)
        os.makedirs(output_dir, exist_ok=True)

    def plot_best_trajectory(self, paths, total_rewards, object_positions_list, step_rewards_list, eval_env):
        if not total_rewards:
            return
        
        max_reward_index = np.argmax(total_rewards)
        best_path = np.array(paths[max_reward_index])
        best_objects = object_positions_list[max_reward_index]
        best_step_rewards = step_rewards_list[max_reward_index]
        
        # Compute noisy observations for the best trajectory
        noisy_path = []
        eval_env.reset()  # Convert seed to Python int
        for pos in best_path:
            eval_env.uav_pos = pos.copy()  # Set UAV position to replay trajectory
            noisy_obs = eval_env._get_obs()  # Get noisy observation
            noisy_path.append(noisy_obs[:2])  # Store noisy UAV position (first two components)
        noisy_path = np.array(noisy_path)
        
        plt.figure(figsize=(8, 8))
        # Plot true trajectory
        plt.plot(best_path[:, 0], best_path[:, 1], 'b-', label='True Trajectory')
        # Plot object and detection zone
        handles, labels = [], []
        for pos in best_objects:
            obj, = plt.plot(pos[0], pos[1], 'ro', markersize=12)
            circle = plt.Circle((pos[0], pos[1]), eval_env.detection_radius, color='g', alpha=0.15)
            plt.gca().add_artist(circle)
            if 'Object' not in labels:
                handles.append(obj)
                labels.append('Object')
            if 'Detection Zone' not in labels:
                handles.append(plt.Circle((0, 0), 1, color='g', alpha=0.15))
                labels.append('Detection Zone')
        start, = plt.plot(best_path[0, 0], best_path[0, 1], 'g^', markersize=15, label='Start')
        handles.append(start)
        labels.append('UAV start')
        for i, pos in enumerate(best_path):
            plt.scatter(pos[0], pos[1], c='g', marker='^', alpha=0.3, s=100)
        # Highlight detection step
        for i in range(len(best_step_rewards)):
            if best_step_rewards[i] >= eval_env.detection_reward:
                detect, = plt.plot(best_path[i+1, 0], best_path[i+1, 1], 'y*', markersize=15, label='Detection Step')
                handles.append(detect)
                labels.append('Detection Step')
                break
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(self.map_bounds_low[0], self.map_bounds_high[0])
        plt.ylim(self.map_bounds_low[1], self.map_bounds_high[1])
        noise_text = (f" (Pos Std: {self.noise_std[0]}m, Bearing Std: {self.noise_std[1]}°)"
                      if isinstance(self.noise_std, tuple) else
                      f" (Noise Std: {self.noise_std})")
        plt.title(f"Best Episode Trajectory with Steps and Detection{noise_text}")
        plt.legend(handles, labels)
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(os.path.join(self.output_dir, "best_trajectory_steps.png"))
        plt.close()

    def plot_all_trajectories(self, paths):
        plt.figure(figsize=(8, 8))
        for path in paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'b-', alpha=0.08)
        plt.plot(0.0, 0.0, 'g^', markersize=15, label='UAV Start')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(self.map_bounds_low[0], self.map_bounds_high[0])
        plt.ylim(self.map_bounds_low[1], self.map_bounds_high[1])
        noise_text = (f" (Pos Std: {self.noise_std[0]}m, Bearing Std: {self.noise_std[1]}°)"
                      if isinstance(self.noise_std, tuple) else
                      f" (Noise Std: {self.noise_std})")
        plt.title(f"All UAV Trajectories Across Evaluation Episodes{noise_text}")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(os.path.join(self.output_dir, "all_trajectories_overlay.png"))
        plt.close()

    def plot_detection_stats(self, detected_list):
        if not all(isinstance(x, (int, bool)) and x in (0, 1) for x in detected_list):
            raise ValueError("detected_list must contain only 0s and 1s or booleans")
        
        plt.figure(figsize=(8, 5))
        detected_count = sum(detected_list)
        undetected_count = len(detected_list) - detected_count
        plt.bar(['Detected', 'Not Detected'], [detected_count, undetected_count], color=['green', 'red'])
        plt.xlabel("Detection Status")
        plt.ylabel("Number of Episodes")
        noise_text = (f" (Pos Std: {self.noise_std[0]}m, Bearing Std: {self.noise_std[1]}°)"
                      if isinstance(self.noise_std, tuple) else
                      f" (Noise Std: {self.noise_std})")
        plt.title(f"Episodes with and without Object Detection{noise_text}")
        plt.savefig(os.path.join(self.output_dir, "detection_episodes.png"))
        plt.close()

    def plot_steps_to_detect(self, detected_list, paths, step_rewards_list, eval_env):
        steps_to_detect = []
        for idx, (detected, path, step_rewards) in enumerate(zip(detected_list, paths, step_rewards_list)):
            if detected == 1:
                for i, reward in enumerate(step_rewards):
                    if reward >= eval_env.detection_reward:
                        steps_to_detect.append(i + 1)
                        break
        
        if steps_to_detect:
            plt.figure(figsize=(8, 5))
            plt.hist(steps_to_detect, bins=15, color='orange', alpha=0.7)
            plt.xlabel("Steps to Detect Object")
            plt.ylabel("Number of Episodes")
            noise_text = (f" (Pos Std: {self.noise_std[0]}m, Bearing Std: {self.noise_std[1]}°)"
                          if isinstance(self.noise_std, tuple) else
                          f" (Noise Std: {self.noise_std})")
            plt.title(f"Steps Required to Detect Object{noise_text}")
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "steps_to_detect_object.png"))
            plt.close()

    def plot_policy_visualization(self, total_rewards, object_positions_list, max_reward_index, model, eval_env):
        if not total_rewards:
            return
        best_objects = object_positions_list[max_reward_index]
        x = np.linspace(self.map_bounds_low[0], self.map_bounds_high[0], 25)
        y = np.linspace(self.map_bounds_low[1], self.map_bounds_high[1], 25)
        X, Y = np.meshgrid(x, y)
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if isinstance(eval_env, NoisyDirectionUAVEnv):
                    rel_pos = best_objects[0] - np.array([X[i, j], Y[i, j]])
                    dist = np.linalg.norm(rel_pos)
                    angle = np.arctan2(rel_pos[1], rel_pos[0]) if dist > 1e-6 else 0.0
                    noisy_angle = angle + np.random.normal(0, eval_env.bearing_noise_std)
                    obs = np.array([X[i, j], Y[i, j], np.sin(noisy_angle), np.cos(noisy_angle)], dtype=np.float32)
                    noise = np.zeros_like(obs)
                    noise[:2] = np.random.normal(0, eval_env.position_noise_std, 2)
                    noisy_obs = obs + noise
                    direction = noisy_obs[2:4]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        noisy_obs[2:4] = direction / norm
                    obs = noisy_obs
                elif isinstance(eval_env, NoisyFullInfoUAVEnv):
                    obs = np.array([X[i, j], Y[i, j], best_objects[0][0], best_objects[0][1]], dtype=np.float32)
                    noise = np.random.normal(0, eval_env.noise_std, obs.shape)
                    obs = np.clip(obs, self.map_bounds_low, self.map_bounds_high)  # Apply clipping
                else:
                    raise ValueError("Unsupported environment type for policy visualization")
                action, _ = model.predict(obs, deterministic=True)
                v_next = (1 - eval_env.damping) * np.array([0, 0]) + action * eval_env.max_speed
                U[i, j], V[i, j] = v_next
        plt.figure(figsize=(9, 9))
        plt.quiver(X, Y, U, V, scale=None, scale_units='xy', width=0.003)
        plt.plot(best_objects[0][0], best_objects[0][1], 'ro', markersize=10, label='Object')
        plt.legend()
        plt.xlim(self.map_bounds_low[0], self.map_bounds_high[0])
        plt.ylim(self.map_bounds_low[1], self.map_bounds_high[1])
        noise_text = (f" (Pos Std: {self.noise_std[0]}m, Bearing Std: {self.noise_std[1]}°)"
                      if isinstance(self.noise_std, tuple) else
                      f" (Noise Std: {self.noise_std})")
        plt.title(f"Policy Vector Field{noise_text}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"uav_policy_noise_{self.noise_std[0]}_{self.noise_std[1]}.png"))
        plt.close()

    def move_figures_to_tensorboard(self, model):
        tb_run_dir = model.logger.dir
        dst_dir = os.path.join(tb_run_dir, "_figures")
        if os.path.exists(self.output_dir):
            os.makedirs(dst_dir, exist_ok=True)
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.png'):  # Chỉ di chuyển file .png
                    src_file = os.path.join(self.output_dir, filename)
                    dst_file = os.path.join(dst_dir, filename)
                    shutil.move(src_file, dst_file)
            print(f"All figures have been moved to TensorBoard log folder: {tb_run_dir}")
        else:
            print("No figures found to move.")

class ComparisonVisualizer:
    """Class for creating comparison visualizations across scenarios and algorithms."""
    
    def __init__(self, output_dir="_comparison_plots"):
        self.output_dir = output_dir
        # Load map bounds from config
        env_config = load_config('env_config')['environment']
        self.map_bounds_low = np.array(env_config['map_bounds']['low'], dtype=np.float32)
        self.map_bounds_high = np.array(env_config['map_bounds']['high'], dtype=np.float32)
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_performance_comparison(self, results_dict, metric_name, title, scenarios, algorithms, single_algo=False):
        """Plot a single metric comparison across scenarios and algorithms."""
        plt.figure(figsize=(8, 5))
        if single_algo:
            values = [results_dict[scenario][metric_name] for scenario in scenarios]
            plt.bar(scenarios, values, color=['blue', 'orange', 'green'])
            plt.title(f'{title} (Single Algorithm: PPO)')
        else:
            for algo in algorithms:
                values = [results_dict[scenario][algo][metric_name] for scenario in scenarios if algo in results_dict[scenario]]
                plt.plot(scenarios, values, marker='o', label=algo)
            plt.title(title)
            plt.legend()
        
        plt.xlabel('Noise Levels')
        plt.ylabel(title)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{metric_name}_comparison.png"), dpi=300)
        plt.close()

    def plot_detection_rate_vs_noise(self, results_dict, noise_levels, scenario_name):
        """Plot detection rate (success rate) as a function of position and bearing noise levels."""
        plt.figure(figsize=(8, 5))
        position_noise_values = [noise_levels['position_noise_levels'][level] for level in noise_levels['position_noise_levels']]
        bearing_noise_values = [noise_levels['bearing_noise_levels'][level] for level in noise_levels['bearing_noise_levels']]
        success_rates = [results_dict[level]['success_rate'] for level in noise_levels['position_noise_levels']]
        
        # Plot detection rate vs position noise, with bearing noise as labels
        plt.plot(position_noise_values, success_rates, marker='o', color='blue', label='Detection Rate')
        # for i, (pos_noise, bearing_noise, success_rate) in enumerate(zip(position_noise_values, bearing_noise_values, success_rates)):
        #     plt.annotate(f"Bearing: {bearing_noise}°", (pos_noise, success_rate), xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Noise Standard Deviation for Position (m) and Brearing (°)')
        plt.ylabel('Detection Rate')
        plt.title(f'Detection Rate vs. Noise Levels for {scenario_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'detection_rate_vs_noise.png'), dpi=300)
        plt.close()

    def generate_all_comparisons(self, results_dict, scenario_name, single_algo=False):
        """Generate comparison plots for all metrics across scenarios."""
        scenarios = list(results_dict.keys())
        algorithms = ["PPO"] if single_algo else list({algo for scenario in results_dict.values() for algo in scenario.keys() if isinstance(scenario, dict)})
        
        # Define metrics to compare
        metrics = [
            ('success_rate', 'Success Rate'),
            ('avg_reward', 'Average Reward'),
            ('avg_steps_to_detect', 'Average Steps to Detect'),
            ('coverage_efficiency', 'Coverage Efficiency')
        ]
        
        # Generate individual comparison plots for each metric
        for metric_name, title in metrics:
            self.plot_performance_comparison(results_dict, metric_name, title, scenarios, algorithms, single_algo)
        
        # Generate detection rate vs. noise level plot
        training_config = load_config('training_config')['training']
        noise_levels = {
            'position_noise_levels': training_config['position_noise_levels'],
            'bearing_noise_levels': training_config['bearing_noise_levels']
        }
        self.plot_detection_rate_vs_noise(results_dict, noise_levels, scenario_name)

    def plot_reward_distribution(self, results_dict):
        """Plot reward distribution comparison."""
        scenarios = list(results_dict.keys())
        algorithms = list(results_dict[scenarios[0]].keys()) if scenarios else []
        
        if not algorithms or not scenarios:
            raise ValueError(f"Cannot generate reward distribution plot: empty scenarios or algorithms (scenarios={scenarios}, algorithms={algorithms})")
        
        fig, axes = plt.subplots(len(scenarios), len(algorithms), 
                                figsize=(6 * len(algorithms), 4 * len(scenarios)))
        
        if len(scenarios) == 1 and len(algorithms) == 1:
            axes = [[axes]]
        elif len(scenarios) == 1:
            axes = [axes]
        elif len(algorithms) == 1:
            axes = [[ax] for ax in axes]
        
        for i, scenario in enumerate(scenarios):
            for j, algo in enumerate(algorithms):
                rewards = results_dict[scenario][algo]['rewards']
                axes[i][j].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
                axes[i][j].set_title(f'{scenario} - {algo}')
                axes[i][j].set_xlabel('Episode Reward')
                axes[i][j].set_ylabel('Frequency')
                axes[i][j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'reward_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_detection_time_analysis(self, results_dict):
        """Plot detection time analysis."""
        scenarios = list(results_dict.keys())
        algorithms = list(results_dict[scenarios[0]].keys()) if scenarios else []
        
        if not algorithms or not scenarios:
            raise ValueError(f"Cannot generate detection time analysis plot: empty scenarios or algorithms (scenarios={scenarios}, algorithms={algorithms})")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of detection times
        detection_data = []
        labels = []
        
        for scenario in scenarios:
            for algo in algorithms:
                detection_times = results_dict[scenario][algo]['detection_times']
                if detection_times:
                    detection_data.append(detection_times)
                    labels.append(f'{scenario}\n{algo}')
        
        if detection_data:
            axes[0].boxplot(detection_data, labels=labels)
            axes[0].set_title('Detection Time Distribution')
            axes[0].set_ylabel('Steps to Detect')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
        
        # Success rate vs average detection time scatter plot
        success_rates = []
        avg_detection_times = []
        colors = []
        scenario_algo_labels = []
        
        color_map = {'PPO': 'blue', 'SAC': 'red', 'DQN': 'green'}
        
        for scenario in scenarios:
            for algo in algorithms:
                success_rate = results_dict[scenario][algo]['success_rate']
                avg_detection_time = results_dict[scenario][algo]['avg_steps_to_detect']
                
                success_rates.append(success_rate)
                avg_detection_times.append(avg_detection_time)
                colors.append(color_map.get(algo, 'black'))
                scenario_algo_labels.append(f'{scenario}-{algo}')
        
        scatter = axes[1].scatter(avg_detection_times, success_rates, 
                                c=colors, s=100, alpha=0.7)
        axes[1].set_xlabel('Average Steps to Detect')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_title('Success Rate vs Detection Time')
        axes[1].grid(True, alpha=0.3)
        
        # Add labels to points
        for i, label in enumerate(scenario_algo_labels):
            axes[1].annotate(label, (avg_detection_times[i], success_rates[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'detection_time_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trajectory_heatmap(self, results_dict):
        """Plot trajectory heatmap for each scenario-algorithm combination."""
        scenarios = list(results_dict.keys())
        algorithms = list(results_dict[scenarios[0]].keys()) if scenarios else []
        
        if not algorithms or not scenarios:
            raise ValueError(f"Cannot generate trajectory heatmap plot: empty scenarios or algorithms (scenarios={scenarios}, algorithms={algorithms})")
        
        fig, axes = plt.subplots(len(scenarios), len(algorithms), 
                                figsize=(6 * len(algorithms), 6 * len(scenarios)))
        
        if len(scenarios) == 1 and len(algorithms) == 1:
            axes = [[axes]]
        elif len(scenarios) == 1:
            axes = [axes]
        elif len(algorithms) == 1:
            axes = [[ax] for ax in axes]
        
        for i, scenario in enumerate(scenarios):
            for j, algo in enumerate(algorithms):
                paths = results_dict[scenario][algo]['paths']
                
                # Create heatmap of visited positions
                heatmap_data = np.zeros((50, 50))
                
                for path in paths:
                    for pos in path:
                        # Convert from map_bounds to [0, 49]
                        x_idx = int((pos[0] - self.map_bounds_low[0]) * 0.95 * 50 / (self.map_bounds_high[0] - self.map_bounds_low[0]))
                        y_idx = int((pos[1] - self.map_bounds_low[1]) * 0.95 * 50 / (self.map_bounds_high[1] - self.map_bounds_low[1]))
                        x_idx = max(0, min(49, x_idx))
                        y_idx = max(0, min(49, y_idx))
                        heatmap_data[y_idx, x_idx] += 1
                
                im = axes[i][j].imshow(heatmap_data, cmap='hot', interpolation='nearest')
                axes[i][j].set_title(f'{scenario} - {algo}')
                axes[i][j].set_xlabel('X Position')
                axes[i][j].set_ylabel('Y Position')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i][j], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'trajectory_heatmaps.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, results_dict):
        """Create a summary report of all metrics."""
        scenarios = list(results_dict.keys())
        algorithms = list(results_dict[scenarios[0]].keys()) if scenarios else []
        
        if not algorithms or not scenarios:
            raise ValueError(f"Cannot generate summary report: empty scenarios or algorithms (scenarios={scenarios}, algorithms={algorithms})")
        
        # Create summary table
        summary_data = []
        for scenario in scenarios:
            for algo in algorithms:
                metrics = results_dict[scenario][algo]
                summary_data.append([
                    scenario,
                    algo,
                    f"{metrics['success_rate']:.3f}",
                    f"{metrics['avg_reward']:.2f}",
                    f"{metrics['avg_steps_to_detect']:.1f}",
                    f"{metrics['coverage_efficiency']:.3f}"
                ])
        
        # Create table plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Scenario', 'Algorithm', 'Success Rate', 
                                 'Avg Reward', 'Avg Steps to Detect', 'Coverage Efficiency'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color code the table
        for i in range(1, len(summary_data) + 1):
            for j in range(len(summary_data[0])):
                if j == 2:  # Success rate column
                    value = float(summary_data[i-1][j])
                    color = plt.cm.RdYlGn(value)
                    table[(i, j)].set_facecolor(color)
        
        plt.title('Performance Summary Report', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.output_dir, 'summary_report.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def move_figures_to_tensorboard(self, model):
        """
        Move all generated figures to the TensorBoard log directory.
        
        Args:
            model: Trained RL model with logger
        """
        # Get the TensorBoard run directory
        tb_run_dir = model.logger.dir
        
        # Create a new directory for all generated files
        generated_files_dir = os.path.join(tb_run_dir, "generated_files")
        os.makedirs(generated_files_dir, exist_ok=True)
        
        # List of directories to check for files
        source_dirs = [
            self.output_dir,  # Comparison plots directory
            "_comparison_plots"  # Another potential output directory
        ]
        
        # Move files from each source directory
        for src_dir in source_dirs:
            if os.path.exists(src_dir):
                for filename in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, filename)
                    dst_file = os.path.join(generated_files_dir, filename)
                    
                    # Check if it's a file (not a directory)
                    if os.path.isfile(src_file):
                        try:
                            shutil.move(src_file, dst_file)
                            print(f"Moved {filename} to {generated_files_dir}")
                        except Exception as e:
                            print(f"Error moving {filename}: {e}")
        
        print(f"All figures have been moved to: {generated_files_dir}")