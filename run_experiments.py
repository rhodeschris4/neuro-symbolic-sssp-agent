import os
import json
import time
from pathlib import Path
from main import run_training

# --------------------------------------------------------------------------
# 1. DEFINE THE EXPERIMENTS
# --------------------------------------------------------------------------
# Define a base configuration with all the parameters that will NOT change.
base_config = {
    'num_episodes': 500,
    'max_steps_per_episode': 150,
    'gamma': 0.99,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'eps_decay': 40000,
    'replay_capacity': 10000,
    'batch_size': 128,
    'target_update_freq': 500,
    'planning_N': 75,
    'planning_C': 20.0,
}

# Define a list of experiments. Each entry is a dictionary of the
# parameters you want to change for that specific run.
experiments = [
    # --- Group 1: Baseline and Controls ---
    {
        "experiment_name": "DQN_Only_Baseline",
        "lr_dqn": 1e-4, "lr_wm": 1e-3,
        "planner_intervention_freq": 99999, "forced_exploration_steps": 0,
        "planning_H": 0, "planning_k": 0,
    },
    {
        "experiment_name": "Standard_Guided_Run",
        "lr_dqn": 1e-4, "lr_wm": 1e-3,
        "planner_intervention_freq": 25, "forced_exploration_steps": 5,
        "planning_H": 15, "planning_k": 3,
    },

    # --- Group 2: Planner Frequency vs. Duration ---
    {
        "experiment_name": "Frequent_Short_Guidance",
        "lr_dqn": 1e-4, "lr_wm": 1e-3,
        "planner_intervention_freq": 15, # Very frequent
        "forced_exploration_steps": 3,   # Short guidance
        "planning_H": 15, "planning_k": 3,
    },
    {
        "experiment_name": "Infrequent_Long_Guidance",
        "lr_dqn": 1e-4, "lr_wm": 1e-3,
        "planner_intervention_freq": 50, # Infrequent
        "forced_exploration_steps": 10,  # Long guidance
        "planning_H": 15, "planning_k": 3,
    },
    {
        "experiment_name": "Very_Frequent_Guidance",
        "lr_dqn": 1e-4, "lr_wm": 1e-3,
        "planner_intervention_freq": 10, # Extremely frequent
        "forced_exploration_steps": 5,
        "planning_H": 15, "planning_k": 3,
    },

    # --- Group 3: Learning Rate Sensitivity ---
    {
        "experiment_name": "High_DQN_Learning_Rate",
        "lr_dqn": 5e-4, # Higher LR
        "lr_wm": 1e-3,
        "planner_intervention_freq": 25, "forced_exploration_steps": 5,
        "planning_H": 15, "planning_k": 3,
    },
    {
        "experiment_name": "Low_DQN_Learning_Rate",
        "lr_dqn": 5e-5, # Lower LR
        "lr_wm": 1e-3,
        "planner_intervention_freq": 25, "forced_exploration_steps": 5,
        "planning_H": 15, "planning_k": 3,
    },

    # --- Group 4: World Model Quality ---
    {
        "experiment_name": "Low_Quality_World_Model",
        "lr_dqn": 1e-4,
        "lr_wm": 1e-4, # Lower WM learning rate -> less accurate map
        "planner_intervention_freq": 25, "forced_exploration_steps": 5,
        "planning_H": 15, "planning_k": 3,
    },

    # --- Group 5: Planner Parameter Sensitivity ---
    {
        "experiment_name": "Shallow_Planner",
        "lr_dqn": 1e-4, "lr_wm": 1e-3,
        "planner_intervention_freq": 25, "forced_exploration_steps": 5,
        "planning_H": 5, # Planner has very short lookahead
        "planning_k": 3,
    },
    {
        "experiment_name": "Deep_Planner",
        "lr_dqn": 1e-4, "lr_wm": 1e-3,
        "planner_intervention_freq": 25, "forced_exploration_steps": 5,
        "planning_H": 25, # Planner has very long lookahead
        "planning_k": 3,
    },
]

# --------------------------------------------------------------------------
# 2. RUN THE EXPERIMENT LOOP
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Create a main directory for all experiment results
    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)

    total_experiments = len(experiments)
    for i, experiment_params in enumerate(experiments):
        # --- Setup for the current run ---
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        exp_name = experiment_params["experiment_name"]
        run_name = f"{timestamp}_{exp_name}"
        run_dir = results_dir / run_name
        run_dir.mkdir(exist_ok=True)

        print(f"\n{'='*80}")
        print(f"🚀 STARTING EXPERIMENT {i+1}/{total_experiments}: {exp_name}")
        print(f"{'='*80}")

        # Combine base config with specific experiment params
        current_config = base_config.copy()
        current_config.update(experiment_params)

        start_time = time.time()

        # --- Run the training ---
        final_metrics = run_training(current_config, run_dir)

        end_time = time.time()
        duration_seconds = end_time - start_time

        # --- Save the report for this run ---
        report = {
            "configuration": current_config,
            "final_metrics": final_metrics,
            "training_duration_seconds": round(duration_seconds, 2)
        }

        report_path = run_dir / "report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        print(f"\n✅ FINISHED EXPERIMENT: {exp_name}")
        print(f"📄 Report and graphs saved to: {run_dir}")

    print(f"\n{'='*80}")
    print("All experiments complete.")
    print(f"{'='*80}")
