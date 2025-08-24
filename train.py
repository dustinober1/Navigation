#!/usr/bin/env python3
"""
Train a DQN agent to solve the Unity Banana Collector environment.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from unityagents import UnityEnvironment

from agent import Agent
from prioritized_agent import PrioritizedAgent
import config


def train_dqn(agent, env, brain_name, n_episodes=config.N_EPISODES, max_t=config.MAX_STEPS, 
              eps_start=config.EPS_START, eps_end=config.EPS_END, eps_decay=config.EPS_DECAY,
              solve_score=config.SOLVE_SCORE, checkpoint_path=config.CHECKPOINT_PATH):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
                
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            
        if np.mean(scores_window) >= solve_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), checkpoint_path)
            break
            
    return scores


def test_agent(agent, env, brain_name, checkpoint_path=config.CHECKPOINT_PATH, num_episodes=5):
    """Test the trained agent."""
    # Load the trained weights
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
    
    for i_episode in range(1, num_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        
        while True:
            action = agent.act(state, eps=0.)  # No exploration during testing
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            
            if done:
                break
                
        print(f"Episode {i_episode}: Score = {score}")


def plot_scores(scores, save_path="training_progress.png"):
    """Plot the training scores."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('DQN Training Progress')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--episodes', type=int, default=config.N_EPISODES, help='Number of episodes')
    parser.add_argument('--checkpoint', type=str, default=config.CHECKPOINT_PATH, help='Checkpoint file')
    parser.add_argument('--double-dqn', action='store_true', help='Use Double DQN')
    parser.add_argument('--dueling-dqn', action='store_true', help='Use Dueling DQN')
    parser.add_argument('--prioritized', action='store_true', help='Use Prioritized Experience Replay')
    
    args = parser.parse_args()
    
    # Initialize environment
    env = UnityEnvironment(file_name=config.UNITY_ENV_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # Get environment info
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    
    print(f'Number of agents: {len(env_info.agents)}')
    print(f'State size: {state_size}')
    print(f'Action size: {action_size}')
    
    # Determine which features to use
    double_dqn = args.double_dqn or config.DOUBLE_DQN
    dueling_dqn = args.dueling_dqn or config.DUELING_DQN
    prioritized = args.prioritized or config.PRIORITIZED_REPLAY
    
    # Print configuration
    print(f'\nDQN Configuration:')
    print(f'  Double DQN: {double_dqn}')
    print(f'  Dueling DQN: {dueling_dqn}')
    print(f'  Prioritized Replay: {prioritized}')
    
    # Initialize agent based on configuration
    if prioritized:
        agent = PrioritizedAgent(
            state_size=state_size, action_size=action_size, seed=config.SEED,
            double_dqn=double_dqn, dueling_dqn=dueling_dqn,
            alpha=config.ALPHA, beta=config.BETA, beta_increment=config.BETA_INCREMENT
        )
    else:
        agent = Agent(
            state_size=state_size, action_size=action_size, seed=config.SEED,
            double_dqn=double_dqn, dueling_dqn=dueling_dqn
        )
    
    if args.train:
        print("\nStarting training...")
        scores = train_dqn(agent, env, brain_name, n_episodes=args.episodes, 
                          checkpoint_path=args.checkpoint)
        
        # Save scores
        np.save(config.SCORES_PATH, scores)
        
        # Plot results
        plot_scores(scores)
        
    if args.test:
        print("\nTesting trained agent...")
        test_agent(agent, env, brain_name, checkpoint_path=args.checkpoint)
    
    # Close environment
    env.close()


if __name__ == '__main__':
    main()