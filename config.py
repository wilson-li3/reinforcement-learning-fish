"""
Configuration file for RL Fish Avoidance System

All simulation parameters in one place for easy experimentation.
"""

# ============================================================================
# ARENA SETTINGS
# ============================================================================
ARENA_RADIUS = 300  # Size of the circular arena (matches current simulation)
ARENA_CENTER = (0, 0)  # Center point of the arena

# ============================================================================
# BALL SETTINGS
# ============================================================================
INITIAL_BALL_RADIUS = 15  # Starting size of the ball
BALL_GROWTH_RATE = 2  # How much the ball grows per fish eaten
MAX_BALL_RADIUS = 100  # Maximum ball size (prevents getting too big)
INITIAL_BALL_SPEED = 8.0  # Initial horizontal velocity
INITIAL_BALL_VERTICAL_SPEED = 15.0  # Initial vertical velocity
GRAVITY = 0.3  # Downward acceleration (matches current simulation)

# ============================================================================
# FISH SETTINGS
# ============================================================================
NUM_FISH = 25  # Number of fish in the simulation
FISH_RADIUS = 8  # Size of each fish (visual and collision)
FISH_MAX_SPEED = 5.0  # Maximum speed a fish can move
FISH_MAX_ACCELERATION = 0.5  # How quickly fish can change direction

# ============================================================================
# TRAINING SETTINGS (we'll use these later)
# ============================================================================
MAX_EPISODE_STEPS = 5000  # Maximum steps per episode
OBSERVATION_DIM = 11  # Number of values in observation (we'll explain later)
ACTION_DIM = 2  # Number of values in action [ax, ay]

# ============================================================================
# REWARD SETTINGS (we'll explain these in Phase 4)
# ============================================================================
REWARD_SURVIVAL = 1.0  # Reward for surviving each timestep
REWARD_CAUGHT = -100.0  # Penalty for getting caught by ball
REWARD_DISTANCE_BONUS = 0.1  # Bonus for being far from ball
REWARD_DANGER_PENALTY = -0.5  # Penalty for being very close to ball
REWARD_ACTION_PENALTY = 0.01  # Small penalty for excessive movement

# Distance thresholds for rewards
SAFE_DISTANCE = 150  # Distance considered "safe" for bonus
DANGER_DISTANCE = 50  # Distance considered "dangerous" for penalty
