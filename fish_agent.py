"""
Fish Agent Module

Defines the fish that will learn to avoid the ball using reinforcement learning.

A fish has:
- Position and velocity (physics)
- Observations (what it can "see")
- Actions (how it controls its movement)
"""

import numpy as np
from config import (
    ARENA_RADIUS, ARENA_CENTER,
    FISH_RADIUS, FISH_MAX_SPEED, FISH_MAX_ACCELERATION
)


class Fish:
    """
    A fish agent that can move within the circular arena.

    The fish:
    - Has position and velocity
    - Can accelerate (action from RL policy)
    - Has maximum speed (realistic movement)
    - Can observe its environment (position, ball location, etc.)
    - Stays within the circular boundary
    """

    def __init__(self, initial_pos=None):
        """
        Initialize a fish.

        Args:
            initial_pos: Starting position [x, y]. If None, randomized.
        """
        # Physical properties (set first!)
        self.radius = FISH_RADIUS
        self.max_speed = FISH_MAX_SPEED
        self.max_acceleration = FISH_MAX_ACCELERATION

        # Arena properties (set before using _random_position!)
        self.arena_radius = ARENA_RADIUS
        self.arena_center = np.array(ARENA_CENTER, dtype=np.float64)

        # Position and velocity
        if initial_pos is None:
            # Randomize initial position within arena
            self.position = self._random_position()
        else:
            self.position = np.array(initial_pos, dtype=np.float64)

        # Start with zero velocity
        self.velocity = np.zeros(2, dtype=np.float64)

        # Alive status (for tracking if caught by ball)
        self.alive = True

    def _random_position(self):
        """
        Generate a random position within the circular arena.

        Returns:
            np.array: Random [x, y] position inside the circle
        """
        # Random angle and distance from center
        angle = np.random.uniform(0, 2 * np.pi)
        # Use sqrt for uniform distribution in circle
        distance = np.sqrt(np.random.uniform(0, 1)) * (self.arena_radius - self.radius - 10)

        x = self.arena_center[0] + distance * np.cos(angle)
        y = self.arena_center[1] + distance * np.sin(angle)

        return np.array([x, y], dtype=np.float64)

    def update(self, action):
        """
        Update fish physics for one timestep based on action.

        Args:
            action: np.array of [ax, ay] in range [-1, 1]
                   Represents desired acceleration direction and magnitude

        Process:
        1. Scale action to actual acceleration
        2. Apply acceleration to velocity
        3. Cap velocity at max speed
        4. Update position
        5. Keep fish inside arena boundary
        """
        # Scale action [-1, 1] to actual acceleration
        acceleration = np.array(action, dtype=np.float64) * self.max_acceleration

        # Apply acceleration to velocity
        self.velocity += acceleration

        # Cap velocity at maximum speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        # Update position
        self.position += self.velocity

        # Keep fish inside arena
        self._handle_boundary()

    def _handle_boundary(self):
        """
        Keep fish inside the circular arena.

        If fish goes outside:
        - Push it back to the boundary
        - Reduce velocity in the outward direction (soft collision)
        """
        distance = np.linalg.norm(self.position - self.arena_center)

        if distance + self.radius > self.arena_radius:
            # Calculate normal (direction from center to fish)
            normal = (self.position - self.arena_center) / distance

            # Push fish back to boundary
            self.position = self.arena_center + normal * (self.arena_radius - self.radius)

            # Reduce velocity in outward direction (soft bounce)
            velocity_along_normal = np.dot(self.velocity, normal)
            if velocity_along_normal > 0:  # Moving outward
                self.velocity -= velocity_along_normal * normal * 0.5

    def get_observation(self, ball):
        """
        Get observation vector for this fish.

        This is what the fish "sees" - the information it uses to make decisions.

        Observation (11 values total):
        1-2. Normalized fish position [x/R, y/R]
        3-4. Normalized fish velocity [vx/max_speed, vy/max_speed]
        5-6. Normalized ball position [x/R, y/R]
        7-8. Normalized ball velocity [vx/15, vy/15] (scaled by typical ball speed)
        9. Normalized distance to ball [dist/2R]
        10. Angle to ball (cosine) [-1, 1]
        11. Angle to ball (sine) [-1, 1]

        Args:
            ball: Ball object to observe

        Returns:
            np.array: Observation vector (11 values, all roughly in [-1, 1])
        """
        # Fish position (normalized by arena radius)
        obs_fish_pos = self.position / self.arena_radius

        # Fish velocity (normalized by max speed)
        obs_fish_vel = self.velocity / self.max_speed

        # Ball position (normalized by arena radius)
        obs_ball_pos = ball.position / self.arena_radius

        # Ball velocity (normalized by typical ball speed ~15)
        obs_ball_vel = ball.velocity / 15.0

        # Calculate vector from fish to ball
        to_ball = ball.position - self.position
        distance = np.linalg.norm(to_ball)

        # Distance to ball (normalized by arena diameter)
        obs_distance = distance / (2 * self.arena_radius)

        # Angle to ball (as cos and sin for continuity)
        if distance > 0:
            to_ball_normalized = to_ball / distance
            obs_angle_cos = to_ball_normalized[0]
            obs_angle_sin = to_ball_normalized[1]
        else:
            obs_angle_cos = 0.0
            obs_angle_sin = 0.0

        # Combine all observations
        observation = np.array([
            obs_fish_pos[0], obs_fish_pos[1],      # 1-2: fish position
            obs_fish_vel[0], obs_fish_vel[1],      # 3-4: fish velocity
            obs_ball_pos[0], obs_ball_pos[1],      # 5-6: ball position
            obs_ball_vel[0], obs_ball_vel[1],      # 7-8: ball velocity (KEY for prediction!)
            obs_distance,                           # 9: distance to ball
            obs_angle_cos, obs_angle_sin           # 10-11: angle to ball
        ], dtype=np.float32)

        return observation

    def check_collision(self, ball):
        """
        Check if this fish collides with the ball.

        Args:
            ball: Ball object

        Returns:
            bool: True if collision detected
        """
        distance = np.linalg.norm(self.position - ball.position)
        return distance < (self.radius + ball.radius)

    def reset(self, initial_pos=None):
        """
        Reset fish to initial state (for new episode or after being caught).

        Args:
            initial_pos: Starting position [x, y]. If None, randomized.
        """
        if initial_pos is None:
            self.position = self._random_position()
        else:
            self.position = np.array(initial_pos, dtype=np.float64)

        self.velocity = np.zeros(2, dtype=np.float64)
        self.alive = True

    def get_state(self):
        """
        Get current state of the fish.

        Returns:
            dict: Fish state with position, velocity, and alive status
        """
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'alive': self.alive
        }


# Test the fish physics (run this file directly to see it work)
if __name__ == "__main__":
    from ball_physics import Ball

    print("Testing Fish Physics...")
    print("=" * 50)

    # Create a fish and a ball
    fish = Fish()
    ball = Ball()

    print(f"Initial fish state: {fish.get_state()}")
    print(f"Initial ball state: {ball.get_state()}")

    # Test observation
    print("\nTesting observation:")
    obs = fish.get_observation(ball)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation values: {obs}")
    print(f"(All values should be roughly in [-1, 1] range)")

    # Test movement with random actions
    print("\nSimulating 5 timesteps with random actions:")
    for i in range(5):
        # Random action in [-1, 1]
        action = np.random.uniform(-1, 1, size=2)
        fish.update(action)
        print(f"Step {i+1}: action={action}, pos={fish.position}, vel={fish.velocity}")

    # Test collision detection
    print("\nTesting collision detection:")
    print(f"Distance to ball: {np.linalg.norm(fish.position - ball.position):.1f}")
    print(f"Collision detected: {fish.check_collision(ball)}")

    # Move fish very close to ball
    fish.position = ball.position + np.array([10.0, 0.0])
    print(f"After moving close - Distance: {np.linalg.norm(fish.position - ball.position):.1f}")
    print(f"Collision detected: {fish.check_collision(ball)}")

    print("\n" + "=" * 50)
    print("Fish physics test complete!")
