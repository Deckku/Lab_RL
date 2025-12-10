"""
Test Value Iteration Agent in Grid World
Trains using value iteration and tests the learned policy.
"""

import numpy as np
from environment.grid_world import GridWorld
from agents.value_iteration_agent import ValueIterationAgent
from utils.visualization import Visualizer
import config

def main():
    print("=" * 60)
    print("VALUE ITERATION AGENT TRAINING")
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
    agent = ValueIterationAgent(
        action_space_n=env.action_space_n,
        observation_space_n=env.observation_space_n,
        environment=env,
        gamma=config.DISCOUNT_FACTOR,
        threshold=config.VALUE_ITERATION_THRESHOLD,
        max_iterations=config.VALUE_ITERATION_MAX_ITERATIONS
    )
    
    # Create visualizer
    viz = Visualizer(env)
    
    # Train agent
    print("\nTraining agent with Value Iteration...")
    train_stats = agent.train()
    
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Converged: {train_stats['converged']}")
    print(f"Iterations: {train_stats['iterations']}")
    print(f"Final Delta: {train_stats['final_delta']:.6f}")
    print("=" * 60)
    
    # Test learned policy
    print("\nTesting learned policy...")
    num_test_episodes = 10
    test_rewards = []
    test_steps = []
    test_trajectories = []
    
    for episode in range(num_test_episodes):
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
        
        test_trajectories.append(trajectory)
        test_rewards.append(episode_reward)
        test_steps.append(steps)
        
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    # Print test summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Test Episodes: {num_test_episodes}")
    print(f"Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Average Steps: {np.mean(test_steps):.2f} ± {np.std(test_steps):.2f}")
    print(f"Success Rate: {sum(r > 0 for r in test_rewards) / num_test_episodes * 100:.1f}%")
    print("=" * 60)
    
    # Visualize results
    print("\nGenerating comprehensive visualization...")
    viz.plot_value_iteration_results(agent)
    
    # Show one test trajectory animation
    if len(test_trajectories) > 0:
        print("\nAnimating best episode...")
        best_episode_idx = np.argmax(test_rewards)
        viz.animate_episode(test_trajectories[best_episode_idx])
    
    print("\nValue iteration training and testing complete!")

if __name__ == "__main__":
    main()