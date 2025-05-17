#!/usr/bin/env python3

import argparse
import os
import sys
import time
from typing import Dict, Optional

import torch

from sarsa import SARSA
from environments.taggame.taggame_constants import (DECAY_RATE, DISCOUNT_RATE, HIDDEN_SIZE,
                             LEARNING_RATE, MIN_EPSILON,
                             MODEL_FILE, N_OF_EPISODES,
                             OUTPUT_DIR, POLICY_EPSILON,
                             POLICY_FILE)
from environments.taggame.taggame_models import TagGameQNet, feature_extractor
from environments.taggame.tag_game import TagGame
from policy import EpsilonGreedyPolicy
from value_strategy import TorchValueStrategy


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_taggame() -> None:
    print("Starting TagGame training with SARSA algorithm and neural network...")
    
    ensure_output_dir()
    
    environment = TagGame()
    try:
        environment.initialize()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Make sure the TagGame Java application is running.")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    FEATURE_SIZE = 13
    model = TagGameQNet(FEATURE_SIZE, HIDDEN_SIZE)
    model.to(device)
    
    value_strategy = TorchValueStrategy(model, feature_extractor, LEARNING_RATE)
    value_strategy.initialize(environment)
    
    model_path = os.path.join(OUTPUT_DIR, MODEL_FILE)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print("Successfully loaded model from file.")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting with a new model.")
    else:
        print("No existing model found. Starting with a new model.")
    
    policy = EpsilonGreedyPolicy(value_strategy, POLICY_EPSILON, MIN_EPSILON, DECAY_RATE)
    
    mdp_solver = SARSA(environment, policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, True)
    
    try:
        print(f"Starting policy iteration with neural network on {device}...")
        mdp_solver.policy_iteration()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reinforcement learning experiments")
    parser.add_argument("--env", type=str, default="taggame",
                      choices=["taggame"],
                      help="Environment to run (default: taggame)")
    
    args = parser.parse_args()
    
    if args.env == "taggame":
        train_taggame()
    else:
        print(f"Unknown environment: {args.env}")
        sys.exit(1)


if __name__ == "__main__":
    main()