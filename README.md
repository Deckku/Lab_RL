# Reinforcement Learning Grid World Project

A complete implementation of reinforcement learning algorithms for grid-based navigation environments.

---

## 🎯 Overview

This project demonstrates three fundamental approaches to solving navigation problems in a grid world:
- **Random Baseline**: Random action selection for comparison
- **Value Iteration**: Planning with complete environment knowledge
- **Q-Learning**: Learning through experience without a model

The agent must navigate from a start position to a goal while avoiding obstacles, with the Q-Learning version supporting dynamic goal movement.

---

## 📦 What's Inside

### Core Components

**Environments** (`environment/`)
- `grid_world.py` - Basic navigation environment with fixed goals
- `grid_world_moving_goal.py` - Advanced version with relocating objectives

**Agents** (`agents/`)
- `base_agent.py` - Abstract interface for all agent types
- `random_agent.py` - Baseline using random actions
- `value_iteration_agent.py` - Dynamic programming solver
- `q_learning_agent.py` - Temporal difference learner

**Utilities** (`utils/`)
- `visualization.py` - Plotting and animation tools

**Configuration & Execution**
- `config.py` - All hyperparameters and settings
- `main_random.py` - Run baseline experiments
- `main_value_iteration.py` - Train planning agent
- `main_q_learning.py` - Train learning agent

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy matplotlib seaborn
```

### Running Experiments

**Baseline Performance**
```bash
python main_random.py
```
Executes 50 episodes with random actions. Shows trajectory overlays, performance distributions, and success metrics.

**Optimal Planning**
```bash
python main_value_iteration.py
```
Computes optimal policy through dynamic programming. Displays convergence behavior, state values, and policy arrows.

**Reinforcement Learning**
```bash
python main_q_learning.py
```
Trains for 500 episodes using Q-Learning. Visualizes learning curves, exploration decay, and final performance.

---

## ⚙️ Customization

Edit `config.py` to modify:
```python
GRID_SIZE = 5                    # Environment dimensions
AGENT_START_POS = (0, 0)        # Initial agent location
GOAL_POSITIONS = [(4, 4)]       # Target location(s)
OBSTACLE_POSITIONS = [(1, 1)]   # Blocked cells

GOAL_REWARD = 10.0              # Success reward
OBSTACLE_REWARD = -5.0          # Collision penalty
STEP_REWARD = -0.1              # Movement cost

LEARNING_RATE = 0.1             # Q-Learning alpha
DISCOUNT_FACTOR = 0.9           # Future reward importance
EPSILON_START = 1.0             # Initial exploration
EPSILON_DECAY = 0.995           # Exploration reduction rate

NUM_EPISODES = 500              # Training duration
GOAL_MOVE_INTERVAL = 20         # Steps until goal relocates
```

---

## 📊 Understanding the Algorithms

### Random Agent
- **Strategy**: Uniform random action selection
- **Purpose**: Performance baseline
- **Characteristics**: No learning, purely exploratory

### Value Iteration
- **Type**: Model-based dynamic programming
- **Requirements**: Complete environment knowledge
- **Process**: Iteratively improves value estimates until convergence
- **Output**: Optimal policy for all states
- **Best for**: Small, fully observable environments

### Q-Learning
- **Type**: Model-free temporal difference learning
- **Requirements**: Only environment interaction
- **Process**: Learns action values through trial and error
- **Features**: Epsilon-greedy exploration with decay
- **Best for**: Unknown or complex environments

---

## 📈 Output Examples

### Random Agent
- Multi-panel trajectory visualization
- Reward and step distributions
- Success/failure breakdown
- Statistical summary

### Value Iteration
- Convergence plot (log scale)
- State value heatmap with annotations
- Policy visualization with directional arrows
- Value evolution for sample states
- Test episode results

### Q-Learning
- Episode reward progression with moving average
- Step count trends
- Epsilon decay curve
- Learned value function heatmap
- Policy extraction
- Success rate over time
- Animated test episode with Q-values

---

## 🔬 Experiment Ideas

1. **Difficulty Scaling**: Increase grid size or add more obstacles
2. **Reward Engineering**: Test different reward structures
3. **Hyperparameter Tuning**: Optimize learning rate, discount factor, epsilon decay
4. **Dynamic Challenges**: Reduce goal movement interval
5. **Algorithm Comparison**: Run all three agents with identical settings
6. **Convergence Analysis**: Track learning speed vs environment complexity

---

## 🛠️ Technical Details

**State Representation**: Discrete integer (0 to n²-1)  
**Action Space**: 4 discrete actions (UP, DOWN, LEFT, RIGHT)  
**Episode Termination**: Goal reached, obstacle hit, or max steps  
**Boundary Handling**: Agent remains in place on invalid moves  

**Value Iteration**:
- Bellman optimality equation
- Synchronous updates
- Convergence threshold: 1e-4

**Q-Learning**:
- Update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- Epsilon-greedy policy
- Off-policy learning

---

## 📝 Notes

- All visualizations use matplotlib and automatically display
- Animations may require closing the window to continue execution
- Training progress prints to console every 50 episodes
- Value iteration typically converges in 50-100 iterations for 5x5 grids
- Q-Learning performance improves significantly after ~200 episodes

---

## 🐛 Troubleshooting

**Import Errors**: Ensure all `__init__.py` files exist in subdirectories  
**No Plots Showing**: Check if matplotlib backend is configured  
**Slow Training**: Reduce `NUM_EPISODES` or `GRID_SIZE`  
**Poor Q-Learning Performance**: Increase training episodes or adjust learning rate  

---

## 📚 Learning Resources

This implementation follows standard RL conventions:
- State-action-reward-next_state tuples
- Episodic task structure
- Gymnasium-style API (without the library)
- Modular agent design

Perfect for understanding:
- Model-based vs model-free methods
- Exploration-exploitation tradeoffs
- Value functions and policies
- Convergence behavior
- Hyperparameter effects

---

**Project Status**: Complete and functional  
**Python Version**: 3.7+  
**Dependencies**: numpy, matplotlib, seaborn  
**License**: Educational use
