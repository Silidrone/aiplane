#!/usr/bin/env python3

import argparse
import os
import sys
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple

from sarsa import SARSA
from environments.windy_grid_world.windy_grid_constants import (
    DECAY_RATE, DISCOUNT_RATE, LEARNING_RATE, MIN_EPSILON,
    N_OF_EPISODES, OUTPUT_DIR, POLICY_EPSILON, POLICY_FILE, VALUE_FILE
)
from environments.windy_grid_world.windy_grid_world import WindyGridWorld, UP, DOWN, LEFT, RIGHT
from environments.windy_grid_world.windy_grid_models import WindyGridWorldQNet, feature_extractor
from policy import EpsilonGreedyPolicy
from value_strategy import TorchValueStrategy


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_windy_grid_world() -> None:
    print("Starting Windy Grid World training with PyTorch and SARSA algorithm...")
    
    ensure_output_dir()
    
    environment = WindyGridWorld()
    environment.initialize()
    
    print("Environment layout:")
    environment.print_grid()
    
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    device = torch.device("cpu")
    print("Using device: CPU (CUDA disabled for stability)")
    
    input_size = environment._grid_height * environment._grid_width * len(environment.all_possible_actions())
    hidden_size = 32
    model = WindyGridWorldQNet(input_size, hidden_size)
    model.to(device)
    
    value_strategy = TorchValueStrategy(model, feature_extractor, LEARNING_RATE)
    value_strategy.initialize(environment)
    
    model_path = os.path.join(OUTPUT_DIR, VALUE_FILE)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            print("Successfully loaded model from file.")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting with a new model.")
    else:
        print("No existing model found. Starting with a new model.")
    
    policy = EpsilonGreedyPolicy(value_strategy, POLICY_EPSILON, MIN_EPSILON, DECAY_RATE)
    
    episode_lengths = []
    evaluation_interval = 100
    
    mdp_solver = SARSA(environment, policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, True)
    
    print(f"Starting policy iteration with neural network on {device}...")
    
    try:
        mdp_solver.policy_iteration()
        
        print("Training completed. Evaluating final policy...")
        
        eval_policy = EpsilonGreedyPolicy(value_strategy, 0.0)
        
        n_eval_episodes = 10
        lengths = []
        
        for i in range(n_eval_episodes):
            steps, _ = run_episode(environment, eval_policy)
            lengths.append(steps)
        
        avg_length = sum(lengths) / len(lengths)
        print(f"Average episode length after training: {avg_length:.1f} steps")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An exception occurred during policy iteration: {e}")
    
    try:
        model.to(torch.device("cpu"))
        torch.save(model.state_dict(), model_path)
        print(f"Successfully saved the model to {model_path}.")
    except Exception as e:
        print(f"Failed to save the model: {e}")
    
    visualize_policy(environment, policy)
    
    steps, trajectory = run_episode(environment, policy)
    print(f"\nOptimal path requires {steps} steps (optimal path has 15 steps)")
    visualize_trajectory(environment, trajectory)


def visualize_policy(environment: WindyGridWorld, policy: EpsilonGreedyPolicy) -> None:
    grid_width = environment._grid_width
    grid_height = environment._grid_height
    
    policy_grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
    
    action_symbols = {
        UP: '↑',
        RIGHT: '→',
        DOWN: '↓',
        LEFT: '←'
    }
    
    for r in range(grid_height):
        for c in range(grid_width):
            state = (r, c)
            if environment.is_terminal(state):
                policy_grid[r][c] = 'G'
            else:
                action, _ = policy.greedy_action(state)
                policy_grid[r][c] = action_symbols[action]
    
    print("\nLearned Policy:")
    print('+' + '-' * (grid_width * 2 - 1) + '+')
    for row in policy_grid:
        print('|' + '|'.join(cell for cell in row) + '|')
    print('+' + '-' * (grid_width * 2 - 1) + '+')
    
    print("\nLegend:")
    print("↑ = Up, → = Right, ↓ = Down, ← = Left, G = Goal")


def run_episode(environment: WindyGridWorld, policy: EpsilonGreedyPolicy, 
                max_steps: int = 1000, debug: bool = False) -> Tuple[int, List[Tuple[int, int]]]:
    state = environment.reset()
    steps = 0
    trajectory = [state]
    
    if debug:
        print("\nStarting episode from state:", state)
    
    while not environment.is_terminal(state) and steps < max_steps:
        action, value = policy.greedy_action(state)
        
        if debug:
            print(f"Step {steps+1}: At state {state}, taking action {action} (Q-value: {value:.2f})")
        
        next_state, reward = environment.step(state, action)
        
        if debug:
            print(f"  Result: New state {next_state}, reward: {reward}")
        
        state = next_state
        trajectory.append(state)
        steps += 1
    
    return steps, trajectory


def visualize_trajectory(environment: WindyGridWorld, trajectory: List[Tuple[int, int]]) -> None:
    grid_width = environment._grid_width
    grid_height = environment._grid_height
    
    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
    
    start_r, start_c = environment._start_state
    goal_r, goal_c = environment._goal_state
    grid[start_r][start_c] = 'S'
    grid[goal_r][goal_c] = 'G'
    
    steps = len(trajectory)
    for i in range(1, steps):
        prev_state = trajectory[i-1]
        curr_state = trajectory[i]
        r, c = curr_state
        
        if (r, c) == environment._start_state or (r, c) == environment._goal_state:
            continue
        
        dr = curr_state[0] - prev_state[0]
        dc = curr_state[1] - prev_state[1]
        
        if dr < 0 and dc == 0:
            arrow = '↑'
        elif dr > 0 and dc == 0:
            arrow = '↓'
        elif dr == 0 and dc > 0:
            arrow = '→'
        elif dr == 0 and dc < 0:
            arrow = '←'
        elif dr < 0 and dc > 0:
            arrow = '↗'
        elif dr < 0 and dc < 0:
            arrow = '↖'
        elif dr > 0 and dc > 0:
            arrow = '↘'
        elif dr > 0 and dc < 0:
            arrow = '↙'
        else:
            arrow = 'o'
        
        grid[r][c] = arrow
    
    print("\nOptimal Trajectory:")
    print('+' + '-' * (grid_width * 2 - 1) + '+')
    for row in grid:
        print('|' + '|'.join(cell for cell in row) + '|')
    print('+' + '-' * (grid_width * 2 - 1) + '+')
    print(f"Steps: {steps - 1}")
    
    print("\nTrajectory sequence:")
    for i, (r, c) in enumerate(trajectory):
        status = "Start" if i == 0 else "Goal" if i == len(trajectory) - 1 else f"Step {i}"
        print(f"{status}: ({r}, {c})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Windy Grid World experiments")
    parser.add_argument("--mode", type=str, default="train",
                      choices=["train", "test"],
                      help="Mode to run (default: train)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug output")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_windy_grid_world()
    elif args.mode == "test":
        environment = WindyGridWorld()
        environment.initialize()
        
        device = torch.device("cpu")
        print("Using device: CPU (CUDA disabled for stability)")
        
        input_size = environment._grid_height * environment._grid_width * len(environment.all_possible_actions())
        hidden_size = 32
        model = WindyGridWorldQNet(input_size, hidden_size)
        model.to(device)
        
        value_strategy = TorchValueStrategy(model, feature_extractor, LEARNING_RATE)
        value_strategy.initialize(environment)
        
        model_path = os.path.join(OUTPUT_DIR, VALUE_FILE)
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                print("Successfully loaded model from file.")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Cannot test without a trained model.")
                return
        else:
            print("No existing model found. Please train first.")
            return
        
        policy = EpsilonGreedyPolicy(value_strategy, 0.0)
        
        print("Environment layout:")
        environment.print_grid()
        
        visualize_policy(environment, policy)
        
        print("\nGenerating optimal trajectory from start to goal...")
        steps, trajectory = run_episode(environment, policy, debug=args.debug)
        print(f"Found path with {steps} steps (optimal path has 15 steps)")
        
        visualize_trajectory(environment, trajectory)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()