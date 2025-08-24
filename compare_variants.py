#!/usr/bin/env python3
"""
Compare different DQN variants on the Unity Banana Collector environment.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time
from unityagents import UnityEnvironment

from agent import Agent
from prioritized_agent import PrioritizedAgent
import config


def train_variant(agent, env, brain_name, variant_name, n_episodes=1000, max_t=config.MAX_STEPS, 
                  eps_start=config.EPS_START, eps_end=config.EPS_END, eps_decay=config.EPS_DECAY,
                  solve_score=config.SOLVE_SCORE):
    """Train a specific DQN variant."""
    print(f"\n{'='*50}")
    print(f"Training {variant_name}")
    print(f"{'='*50}")
    
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    start_time = time.time()
    solved_episode = None
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
                
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            
        if np.mean(scores_window) >= solve_score and solved_episode is None:
            solved_episode = i_episode - 100
            print(f'\n{variant_name} solved in {solved_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            
        if solved_episode is not None and i_episode >= solved_episode + 200:
            break
    
    training_time = time.time() - start_time
    final_score = np.mean(scores_window)
    
    print(f"\n{variant_name} Results:")
    print(f"  Episodes to solve: {solved_episode if solved_episode else 'Not solved'}")
    print(f"  Final average score: {final_score:.2f}")
    print(f"  Training time: {training_time:.1f} seconds")
    
    return scores, solved_episode, final_score, training_time


def compare_dqn_variants():
    """Compare different DQN variants."""
    # Initialize environment
    env = UnityEnvironment(file_name=config.UNITY_ENV_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    
    print(f'Environment initialized:')
    print(f'  State size: {state_size}')
    print(f'  Action size: {action_size}')
    
    # Define variants to compare
    variants = [
        {
            'name': 'Standard DQN',
            'agent': Agent(state_size, action_size, config.SEED),
            'color': 'blue'
        },
        {
            'name': 'Double DQN',
            'agent': Agent(state_size, action_size, config.SEED, double_dqn=True),
            'color': 'red'
        },
        {
            'name': 'Dueling DQN',
            'agent': Agent(state_size, action_size, config.SEED, dueling_dqn=True),
            'color': 'green'
        },
        {
            'name': 'Double + Dueling DQN',
            'agent': Agent(state_size, action_size, config.SEED, double_dqn=True, dueling_dqn=True),
            'color': 'orange'
        },
        {
            'name': 'Prioritized Experience Replay',
            'agent': PrioritizedAgent(state_size, action_size, config.SEED),
            'color': 'purple'
        },
        {
            'name': 'Rainbow (All features)',
            'agent': PrioritizedAgent(state_size, action_size, config.SEED, double_dqn=True, dueling_dqn=True),
            'color': 'gold'
        }
    ]
    
    results = {}
    all_scores = {}
    
    # Train each variant
    for variant in variants:
        scores, solved_ep, final_score, train_time = train_variant(
            variant['agent'], env, brain_name, variant['name']
        )
        
        results[variant['name']] = {
            'solved_episode': solved_ep,
            'final_score': final_score,
            'training_time': train_time
        }
        all_scores[variant['name']] = {
            'scores': scores,
            'color': variant['color']
        }
    
    env.close()
    
    # Plot comparison
    plot_comparison(all_scores, results)
    
    # Print summary table
    print_summary_table(results)


def plot_comparison(all_scores, results):
    """Plot training curves for all variants."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training curves
    plt.subplot(2, 2, 1)
    for name, data in all_scores.items():
        scores = data['scores']
        # Calculate rolling average
        window_size = 100
        if len(scores) >= window_size:
            rolling_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(scores)), rolling_avg, 
                    color=data['color'], label=name, linewidth=2)
    
    plt.axhline(y=13, color='black', linestyle='--', alpha=0.7, label='Solve threshold')
    plt.xlabel('Episode')
    plt.ylabel('Score (100-episode average)')
    plt.title('Training Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Episodes to solve
    plt.subplot(2, 2, 2)
    solved_episodes = [results[name]['solved_episode'] for name in results.keys() if results[name]['solved_episode'] is not None]
    variant_names = [name for name in results.keys() if results[name]['solved_episode'] is not None]
    colors = [all_scores[name]['color'] for name in variant_names]
    
    bars = plt.bar(range(len(variant_names)), solved_episodes, color=colors)
    plt.xlabel('DQN Variant')
    plt.ylabel('Episodes to Solve')
    plt.title('Sample Efficiency Comparison')
    plt.xticks(range(len(variant_names)), variant_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, episodes in zip(bars, solved_episodes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{episodes}', ha='center', va='bottom')
    
    # Plot 3: Final performance
    plt.subplot(2, 2, 3)
    final_scores = [results[name]['final_score'] for name in results.keys()]
    colors = [all_scores[name]['color'] for name in results.keys()]
    
    bars = plt.bar(range(len(results)), final_scores, color=colors)
    plt.axhline(y=13, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('DQN Variant')
    plt.ylabel('Final Average Score')
    plt.title('Final Performance Comparison')
    plt.xticks(range(len(results)), list(results.keys()), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, final_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{score:.1f}', ha='center', va='bottom')
    
    # Plot 4: Training time
    plt.subplot(2, 2, 4)
    training_times = [results[name]['training_time']/60 for name in results.keys()]  # Convert to minutes
    
    bars = plt.bar(range(len(results)), training_times, color=colors)
    plt.xlabel('DQN Variant')
    plt.ylabel('Training Time (minutes)')
    plt.title('Training Time Comparison')
    plt.xticks(range(len(results)), list(results.keys()), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, time_min in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time_min:.1f}m', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dqn_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_table(results):
    """Print a summary table of all results."""
    print(f"\n{'='*80}")
    print("DQN VARIANTS COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Variant':<25} {'Episodes to Solve':<18} {'Final Score':<12} {'Training Time':<15}")
    print(f"{'-'*25} {'-'*18} {'-'*12} {'-'*15}")
    
    for name, data in results.items():
        episodes = data['solved_episode'] if data['solved_episode'] else "Not solved"
        score = f"{data['final_score']:.2f}"
        time_str = f"{data['training_time']/60:.1f}m"
        
        print(f"{name:<25} {str(episodes):<18} {score:<12} {time_str:<15}")
    
    # Find best performers
    solved_variants = {name: data for name, data in results.items() if data['solved_episode'] is not None}
    
    if solved_variants:
        fastest = min(solved_variants.items(), key=lambda x: x[1]['solved_episode'])
        best_score = max(results.items(), key=lambda x: x[1]['final_score'])
        fastest_training = min(results.items(), key=lambda x: x[1]['training_time'])
        
        print(f"\n{'='*80}")
        print("BEST PERFORMERS:")
        print(f"  Most Sample Efficient: {fastest[0]} ({fastest[1]['solved_episode']} episodes)")
        print(f"  Highest Final Score:   {best_score[0]} ({best_score[1]['final_score']:.2f})")
        print(f"  Fastest Training:      {fastest_training[0]} ({fastest_training[1]['training_time']/60:.1f}m)")


def main():
    parser = argparse.ArgumentParser(description='Compare DQN variants')
    parser.add_argument('--episodes', type=int, default=1000, help='Max episodes per variant')
    
    args = parser.parse_args()
    
    print("Starting DQN variants comparison...")
    print(f"Each variant will train for up to {args.episodes} episodes")
    print("Training will stop early if solved + 200 additional episodes")
    
    compare_dqn_variants()


if __name__ == '__main__':
    main()