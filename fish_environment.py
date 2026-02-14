"""
Fish Environment Module

Gymnasium environment where fish learn to avoid a bouncing ball.

This is a multi-agent environment:
- Multiple fish (all using the same policy)
- One ball that bounces and grows
- Circular arena boundary

The environment handles:
- Physics updates (ball and fish)
- Collision detection
- Reward calculation
- Episode termination
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ball_physics import Ball
from fish_agent import Fish
from config import (
    NUM_FISH, OBSERVATION_DIM, ACTION_DIM,
    MAX_EPISODE_STEPS,
    REWARD_SURVIVAL, REWARD_CAUGHT,
    REWARD_DISTANCE_BONUS, REWARD_DANGER_PENALTY,
    REWARD_ACTION_PENALTY,
    SAFE_DISTANCE, DANGER_DISTANCE,
    INITIAL_BALL_SPEED, INITIAL_BALL_VERTICAL_SPEED
)


class FishEnvironment(gym.Env):
    """
    Gymnasium environment for fish avoiding a bouncing ball.

    Observation Space: Box(11,) for each fish
        - Fish position (normalized)
        - Fish velocity (normalized)
        - Ball position (normalized)
        - Ball velocity (normalized)
        - Distance to ball
        - Angle to ball (cos, sin)

    Action Space: Box(2,) for each fish
        - [ax, ay]: Acceleration in x and y directions, range [-1, 1]

    Rewards:
        - +1 for each timestep survived
        - -100 for getting caught by ball
        - +0.1 bonus for being far from ball
        - -0.5 penalty for being very close to ball
        - Small penalty for excessive movement
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, num_fish=NUM_FISH):
        """
        Initialize the fish environment.

        Args:
            num_fish: Number of fish in the environment
        """
        super().__init__()

        self.num_fish = num_fish

        # Define observation and action spaces (per fish)
        # Observations: 11 continuous values in roughly [-1, 1]
        self.observation_space = spaces.Box(
            low=-2.0,  # Some values might slightly exceed [-1, 1]
            high=2.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32
        )

        # Actions: 2 continuous values in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32
        )

        # Create ball and fish
        self.ball = None
        self.fish_list = None

        # Episode tracking
        self.timestep = 0
        self.total_fish_caught = 0
        self.episode_rewards = None

        # Initialize
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.

        Returns:
            observation: Initial observation for the first fish
            info: Additional information dictionary
        """
        # Handle seeding
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset ball to initial state
        self.ball = Ball(
            initial_pos=np.array([0.0, -100.0]),
            initial_velocity=np.array([INITIAL_BALL_SPEED, INITIAL_BALL_VERTICAL_SPEED])
        )

        # Reset all fish to random positions
        self.fish_list = [Fish() for _ in range(self.num_fish)]

        # Reset episode tracking
        self.timestep = 0
        self.total_fish_caught = 0
        self.episode_rewards = np.zeros(self.num_fish, dtype=np.float32)

        # Return initial observation for first fish (Gymnasium expects single obs)
        observation = self.fish_list[0].get_observation(self.ball)
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one timestep of the environment.

        For multi-agent: This handles ONE fish at a time (called in a loop by training code).
        For simplicity, we'll make this work with vectorized environments later.

        Args:
            action: Action for one fish [ax, ay]

        Returns:
            observation: Next observation
            reward: Reward for this timestep
            terminated: Whether episode ended
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        # Update ball physics
        self.ball.update()

        # For now, we'll handle the primary fish (index 0)
        # In vectorized version, we'll handle all fish
        fish = self.fish_list[0]

        # Apply action to fish
        fish.update(action)

        # Check collision
        caught = fish.check_collision(self.ball)

        # Calculate reward
        reward = self._calculate_reward(fish, action, caught)

        # Handle collision (fish respawns, ball grows)
        if caught:
            self.ball.grow()
            fish.reset()  # Respawn at random position
            self.total_fish_caught += 1

        # Update timestep
        self.timestep += 1

        # Check termination conditions
        terminated = False  # Individual fish don't end episode when caught
        truncated = self.timestep >= MAX_EPISODE_STEPS

        # Get next observation
        observation = fish.get_observation(self.ball)

        # Info dict
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def step_all_fish(self, actions):
        """
        Execute one timestep for ALL fish simultaneously.

        This is the multi-agent version we'll use for training.

        Args:
            actions: List of actions, one per fish [[ax1,ay1], [ax2,ay2], ...]

        Returns:
            observations: List of observations, one per fish
            rewards: List of rewards, one per fish
            dones: List of done flags, one per fish
            truncated: Whether episode was truncated (same for all)
            info: Additional information
        """
        # Update ball physics (once per timestep, not per fish)
        self.ball.update()

        observations = []
        rewards = []
        dones = []

        # Update each fish
        for i, (fish, action) in enumerate(zip(self.fish_list, actions)):
            # Apply action to fish
            fish.update(action)

            # Check collision
            caught = fish.check_collision(self.ball)

            # Calculate reward
            reward = self._calculate_reward(fish, action, caught)
            rewards.append(reward)

            # Handle collision
            if caught:
                self.ball.grow()
                fish.reset()  # Respawn
                self.total_fish_caught += 1
                dones.append(True)  # This fish's sub-episode ends
            else:
                dones.append(False)

            # Get observation for next timestep
            obs = fish.get_observation(self.ball)
            observations.append(obs)

        # Update timestep
        self.timestep += 1

        # Episode truncation (applies to all fish)
        truncated = self.timestep >= MAX_EPISODE_STEPS

        # Info
        info = self._get_info()

        return observations, rewards, dones, truncated, info

    def _calculate_reward(self, fish, action, caught):
        """
        Calculate reward for one fish for one timestep.

        Args:
            fish: Fish object
            action: Action taken [ax, ay]
            caught: Whether fish was caught this timestep

        Returns:
            float: Reward value
        """
        if caught:
            # Terminal penalty for getting caught
            return REWARD_CAUGHT

        # Base survival reward
        reward = REWARD_SURVIVAL

        # Distance-based shaping rewards
        distance = np.linalg.norm(fish.position - self.ball.position)

        if distance > SAFE_DISTANCE:
            # Bonus for being far away (safe)
            reward += REWARD_DISTANCE_BONUS
        elif distance < DANGER_DISTANCE:
            # Penalty for being very close (dangerous)
            reward += REWARD_DANGER_PENALTY  # Note: this is negative

        # Small penalty for excessive movement (encourages efficiency)
        action_magnitude = np.linalg.norm(action)
        reward -= REWARD_ACTION_PENALTY * action_magnitude

        return reward

    def _get_info(self):
        """
        Get additional information about the environment state.

        Returns:
            dict: Info dictionary
        """
        return {
            'timestep': self.timestep,
            'total_fish_caught': self.total_fish_caught,
            'ball_radius': self.ball.radius,
            'ball_position': self.ball.position.copy(),
            'num_fish_alive': sum(1 for fish in self.fish_list if fish.alive)
        }

    def get_all_observations(self):
        """
        Get observations for all fish.

        Returns:
            np.array: Array of observations, shape (num_fish, obs_dim)
        """
        observations = [fish.get_observation(self.ball) for fish in self.fish_list]
        return np.array(observations, dtype=np.float32)

    def render(self):
        """
        Render the environment (for visualization).
        Not implemented here - we'll use matplotlib visualization separately.
        """
        pass


# Wrapper for single-agent training (trains one fish at a time)
class SingleFishEnv(gym.Env):
    """
    Wrapper that exposes a single fish for standard RL training.

    The same policy will be used for all fish, but we train on one fish's
    experience at a time. This makes it compatible with standard RL libraries.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()

        # Create the multi-agent environment
        self.env = FishEnvironment()

        # Expose the observation and action spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Track which fish we're controlling
        self.current_fish_idx = 0

    def reset(self, seed=None, options=None):
        """Reset environment and return observation for first fish."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.current_fish_idx = 0
        return obs, info

    def step(self, action):
        """
        Step the environment with action for current fish.

        We'll cycle through fish or just control fish 0 for simplicity.
        """
        # Update ball physics
        self.env.ball.update()

        # Get current fish
        fish = self.env.fish_list[self.current_fish_idx]

        # Apply action
        fish.update(action)

        # Check collision
        caught = fish.check_collision(self.env.ball)

        # Calculate reward
        reward = self.env._calculate_reward(fish, action, caught)

        # Handle collision
        if caught:
            self.env.ball.grow()
            fish.reset()
            self.env.total_fish_caught += 1

        # Update timestep
        self.env.timestep += 1

        # Check termination
        terminated = False
        truncated = self.env.timestep >= MAX_EPISODE_STEPS

        # Get observation
        observation = fish.get_observation(self.env.ball)

        # Info
        info = self.env._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        return self.env.render()


# Test the environment
if __name__ == "__main__":
    print("Testing Fish Environment...")
    print("=" * 60)

    # Create environment
    env = SingleFishEnv()
    print(f"Created environment with {env.env.num_fish} fish")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")

    # Run a few random steps
    print("\nRunning 10 random timesteps:")
    total_reward = 0
    for i in range(10):
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        done = terminated or truncated

        print(f"Step {i+1}: reward={reward:.2f}, timestep={info['timestep']}, "
              f"fish_caught={info['total_fish_caught']}, ball_radius={info['ball_radius']:.1f}")

        if done:
            print("Episode ended!")
            break

    print(f"\nTotal reward after 10 steps: {total_reward:.2f}")

    # Test multi-agent step
    print("\n" + "=" * 60)
    print("Testing multi-agent step_all_fish()...")

    env = FishEnvironment(num_fish=5)  # Smaller for testing
    obs, info = env.reset()

    # Random actions for all fish
    actions = [env.action_space.sample() for _ in range(5)]
    observations, rewards, dones, truncated, info = env.step_all_fish(actions)

    print(f"Observations shape: {len(observations)} x {observations[0].shape}")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    print(f"Info: {info}")

    print("\n" + "=" * 60)
    print("Environment test complete!")
