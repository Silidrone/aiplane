import os
import torch
import time
import pygame
import argparse

import sys
sys.path.append('/home/silidrone/silidev/aiplane_py')  # Add root directory to path

from sarsa import SARSA
from constants import (
    DECAY_RATE, DISCOUNT_RATE, ENABLE_RENDERING, LEARNING_RATE, MIN_EPSILON,
    N_OF_EPISODES, OUTPUT_DIR, POLICY_EPSILON, MODEL_FILE, HIDDEN_SIZE
)
from taggame import TagGame
from models import TagGameQNet, feature_extractor
from policy import EpsilonGreedyPolicy
from value_strategy import TorchValueStrategy

MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_FILE)

def setup_training():
    """Initialize the environment, model, and training components"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    environment = TagGame(render=ENABLE_RENDERING)
    environment.initialize()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_size = 14  # Number of features in feature_extractor
    
    model = TagGameQNet(input_size, HIDDEN_SIZE)
    model.to(device)
    
    value_strategy = TorchValueStrategy(model, feature_extractor, LEARNING_RATE, device)
    value_strategy.initialize(environment)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting with a new model.")
    else:
        print("No existing model found. Starting with a new model.")
    
    policy = EpsilonGreedyPolicy(value_strategy, POLICY_EPSILON, MIN_EPSILON, DECAY_RATE)
    mdp_solver = SARSA(environment, policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, True)
    
    return environment, model, value_strategy, policy, mdp_solver

def train(mdp_solver, model):
    print(f"Starting training with neural network...")
    
    try:
        start_time = time.time()
        
        mdp_solver.policy_iteration()
        
        training_time = time.time() - start_time

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved model. Training completed in {training_time:.2f} seconds.")
    except Exception as e:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved model. Training stopped because: {e}")

def evaluate(environment, policy, n_episodes=100):
    print(f"Evaluating policy for {n_episodes} episodes...")
    
    if hasattr(policy.value_strategy, 'q_network'):
        policy.value_strategy.q_network.eval()
    
    episode_lengths = []
    episode_rewards = []
    
    for i in range(n_episodes):
        state = environment.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = policy.greedy_action(state)
            next_state, reward = environment.step(state, action)
            total_reward += reward
            steps += 1
            state = next_state
            done = environment.is_terminal(state)
            
            if pygame.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
        
        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        
        print(f"Episode {i+1} - Steps: {steps}, Total Reward: {total_reward:.2f}")
    
    avg_length = sum(episode_lengths) / len(episode_lengths)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Evaluation Summary:")
    print(f"Average Episode Length: {avg_length:.2f} steps")
    print(f"Average Episode Reward: {avg_reward:.2f}")
    
def main():
    parser = argparse.ArgumentParser(description='TagGame RL Training')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate'],
                        help='Mode to run: train or evaluate')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
                        
    args = parser.parse_args()
    
    environment, model, value_strategy, policy, mdp_solver = setup_training()
    try:
        if args.mode == 'train':
            train(mdp_solver, model)
        elif args.mode == 'evaluate':
            evaluate(environment, policy, args.episodes)
        
        if environment.render_enabled:
            environment.close()
    except Exception as e:
        print(f"Stopped: {e}")


if __name__ == "__main__":
    main()