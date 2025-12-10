"""
Test Random Agent in Grid World
Runs multiple episodes and visualizes performance.
"""

import numpy as np
from environment.grid_world import GridWorld
from agents.random_agent import RandomAgent
from utils.visualization import Visualizer
import config

def main():
    print("=" * 60)
    print("RANDOM AGENT TESTING")
    print("=" * 60)
    
    # Create environment
    env = GridWorld(
        grid_size=config.GRID_SIZE,
        agent_start_pos=config.AGENT_START_POS,
        goal_positions=config.GOAL_POSITIONS,
        obstacle_positions=config.OBSTACLE_POSITIONS,
        goal_reward=config.GOAL_REWARD,
        obstacle_reward=config.OBSTACLE_REWARD,
        step_reward=config.STEP_REWARD
    )
    
    # Create agent
    agent = RandomAgent(
        action_space_n=env.action_space_n,
        observation_space_n=env.observation_space_n
    )
    
    # Create visualizer
    viz = Visualizer(env)
    
    # Run multiple episodes
    num_episodes = 50
    trajectories = []
    rewards = []
    steps_list = []
    
    print(f"\nRunning {num_episodes} episodes with random agent...")
    
    for episode in range(num_episodes):
        state = env.reset()
        trajectory = [env.agent_pos.copy()]
        episode_reward = 0
        steps = 0
        
        for step in range(config.MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            trajectory.append(env.agent_pos.copy())
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        trajectories.append(trajectory)
        rewards.append(episode_reward)
        steps_list.append(steps)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}")
    print(f"Success Rate: {sum(r > 0 for r in rewards) / num_episodes * 100:.1f}%")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print("=" * 60)
    
    # Visualize results
    print("\nGenerating comprehensive visualization...")
    viz.plot_random_agent_results(trajectories, rewards, steps_list)
    
    print("\nRandom agent testing complete!")

if __name__ == "__main__":
    main()