# Configuration parameters for Grid World RL

# Grid world settings
GRID_SIZE = 5
AGENT_START_POS = (0, 0)
GOAL_POSITIONS = [(4, 4)]
OBSTACLE_POSITIONS = [(1, 1), (2, 2), (3, 1)]

# Reward structure
GOAL_REWARD = 10.0
OBSTACLE_REWARD = -5.0
STEP_REWARD = -0.1

# Learning parameters
DISCOUNT_FACTOR = 0.9  # Gamma
LEARNING_RATE = 0.1    # Alpha for Q-Learning
EPSILON_START = 1.0    # Initial exploration rate
EPSILON_MIN = 0.01     # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate per episode

# Training settings
MAX_STEPS_PER_EPISODE = 100
NUM_EPISODES = 500
VALUE_ITERATION_THRESHOLD = 1e-4
VALUE_ITERATION_MAX_ITERATIONS = 1000

# Moving goal settings
GOAL_MOVE_INTERVAL = 20  # Steps before goal moves
GOAL_MOVE_ENABLED = True

# Visualization settings
FIGURE_SIZE = (15, 10)
DPI = 100
ANIMATION_INTERVAL = 200  # ms