import os
import torch
import matplotlib.pyplot as plt
import time
import pygame
import argparse
from datetime import datetime

import sys
sys.path.append('/home/silidrone/silidev/aiplane_py')  # Add root directory to path

from sarsa import SARSA
from constants import (
    DECAY_RATE, DISCOUNT_RATE, ENABLE_RENDERING, LEARNING_RATE, MIN_EPSILON,
    N_OF_EPISODES, OUTPUT_DIR, POLICY_EPSILON, MODEL_FILE, HIDDEN_SIZE
)
from taggame import TagGame
from models import TagGameQNet, feature_extractor, set_device, state_to_readable
from policy import EpsilonGreedyPolicy
from value_strategy import TorchValueStrategy

def setup_training():
    """Initialize the environment, model, and training components"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    environment = TagGame(render=ENABLE_RENDERING)
    environment.initialize()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_device(device)
    
    input_size = 14  # Number of features in feature_extractor
    
    model = TagGameQNet(input_size, HIDDEN_SIZE)
    model.to(device)
    
    value_strategy = TorchValueStrategy(model, feature_extractor, LEARNING_RATE)
    value_strategy.initialize(environment)
    
    model_path = os.path.join(OUTPUT_DIR, MODEL_FILE)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print(f"Successfully loaded model from {model_path}")
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
        print(f"Training completed in {training_time:.2f} seconds")
        
        model_path = os.path.join(OUTPUT_DIR, MODEL_FILE)
        try:
            torch.save(model.state_dict(), model_path)
            print(f"Successfully saved model to {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
        # Save the model on interruption
        model_path = os.path.join(OUTPUT_DIR, MODEL_FILE)
        try:
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path} after interruption")
        except Exception as e:
            print(f"Failed to save model: {e}")
    
    except Exception as e:
        print(f"An error occurred during training: {e}")

def evaluate(environment, policy, n_episodes=10):
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
        
        print(f"Episode {i+1} starting state: {state_to_readable(state)}")
        
        while not done and steps < 1000:  # Max 1000 steps per episode
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
            
            if environment.render_enabled:
                pygame.time.delay(50)  # 50ms delay
        
        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        
        print(f"Episode {i+1} - Steps: {steps}, Total Reward: {total_reward:.2f}")
    
    avg_length = sum(episode_lengths) / len(episode_lengths)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Evaluation Summary:")
    print(f"Average Episode Length: {avg_length:.2f} steps")
    print(f"Average Episode Reward: {avg_reward:.2f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, n_episodes+1), episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Length (steps)')
    plt.title('Episode Lengths')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(1, n_episodes+1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    print(f"Evaluation plot saved to {plot_path}")
    
    plt.show()

def run_interactive(environment, policy):
    """Run the environment with the policy and allow user interaction"""
    print("Running interactive mode. Press ESC to exit.")
    
    state = environment.reset()
    done = False
    total_reward = 0
    steps = 0
    
    if not environment.render_enabled:
        print("Cannot run interactive mode without rendering. Exiting.")
        return
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = environment.reset()
                    done = False
                    total_reward = 0
                    steps = 0
                    print("Environment reset")
        
        if not done:
            action, _ = policy.greedy_action(state)
            
            next_state, reward = environment.step(state, action)
            total_reward += reward
            steps += 1
            
            print(f"Step {steps}: Action: {action}, Reward: {reward:.4f}, Total: {total_reward:.4f}")
            
            state = next_state
            done = environment.is_terminal(state)
            
            if done:
                print(f"Episode ended after {steps} steps with total reward {total_reward:.4f}")
                print("Press R to reset or ESC to exit")
        
        pygame.time.delay(50)
    
    environment.close()

def main():
    parser = argparse.ArgumentParser(description='TagGame RL Training')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate', 'interactive'],
                        help='Mode to run: train, evaluate, or interactive')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
                        
    args = parser.parse_args()
    
    environment, model, value_strategy, policy, mdp_solver = setup_training()
    
    if args.mode == 'train':
        train(mdp_solver, model)
    elif args.mode == 'evaluate':
        evaluate(environment, policy, args.episodes)
    elif args.mode == 'interactive':
        run_interactive(environment, policy)
    
    if environment.render_enabled:
        environment.close()

if __name__ == "__main__":
    main()