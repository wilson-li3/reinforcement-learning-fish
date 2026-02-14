"""
Ball Physics Module

Handles the bouncing ball simulation with gravity and circular boundary collision.
Extracted from bouncing_ball_matplotlib.py for reusability.
"""

import numpy as np
from config import (
    ARENA_RADIUS, ARENA_CENTER,
    INITIAL_BALL_RADIUS, BALL_GROWTH_RATE, MAX_BALL_RADIUS,
    GRAVITY
)


class Ball:
    """
    A ball that bounces within a circular arena with gravity.

    The ball:
    - Falls due to gravity
    - Bounces off the circular boundary with perfect elastic collision
    - Can grow in size when it "eats" fish
    """

    def __init__(self, initial_pos=None, initial_velocity=None):
        """
        Initialize the ball.

        Args:
            initial_pos: Starting position [x, y]. Defaults to [0, -100]
            initial_velocity: Starting velocity [vx, vy]. Defaults to [8, 15]
        """
        # Position and velocity
        if initial_pos is None:
            self.position = np.array([0.0, -100.0], dtype=np.float64)
        else:
            self.position = np.array(initial_pos, dtype=np.float64)

        if initial_velocity is None:
            self.velocity = np.array([8.0, 15.0], dtype=np.float64)
        else:
            self.velocity = np.array(initial_velocity, dtype=np.float64)

        # Size
        self.radius = INITIAL_BALL_RADIUS

        # Arena properties
        self.arena_radius = ARENA_RADIUS
        self.arena_center = np.array(ARENA_CENTER, dtype=np.float64)

    def update(self):
        """
        Update ball physics for one timestep.

        Process:
        1. Apply gravity (downward acceleration)
        2. Update position based on velocity
        3. Check for collision with circular boundary
        4. If collision, bounce (reflect velocity) and push ball back inside
        """
        # Apply gravity to velocity
        self.velocity[1] -= GRAVITY

        # Update position
        self.position += self.velocity

        # Check collision with circular boundary
        self._handle_boundary_collision()

    def _handle_boundary_collision(self):
        """
        Handle collision with the circular arena boundary.

        Uses vector reflection for perfect elastic collision:
        - Calculate normal vector (from center to ball)
        - Reflect velocity across this normal
        - Push ball back inside if overlapping boundary
        """
        # Calculate distance from ball center to arena center
        distance = np.linalg.norm(self.position - self.arena_center)

        # Check if ball has collided with boundary
        if distance + self.radius >= self.arena_radius:
            # Calculate normal vector (direction from center to ball)
            normal = (self.position - self.arena_center) / distance

            # Calculate velocity component along normal
            velocity_along_normal = np.dot(self.velocity, normal)

            # Reflect velocity (perfectly elastic collision)
            # Formula: v_new = v_old - 2 * (v · n) * n
            self.velocity -= 2 * velocity_along_normal * normal

            # Push ball back inside arena (prevent sticking to boundary)
            overlap = (distance + self.radius) - self.arena_radius
            self.position -= overlap * normal

    def grow(self):
        """
        Increase ball size (called when ball eats a fish).
        Ball grows by BALL_GROWTH_RATE up to MAX_BALL_RADIUS.
        """
        self.radius = min(self.radius + BALL_GROWTH_RATE, MAX_BALL_RADIUS)

    def reset(self, initial_pos=None, initial_velocity=None):
        """
        Reset ball to initial state.

        Args:
            initial_pos: Starting position [x, y]. Defaults to [0, -100]
            initial_velocity: Starting velocity [vx, vy]. Defaults to [8, 15]
        """
        if initial_pos is None:
            self.position = np.array([0.0, -100.0], dtype=np.float64)
        else:
            self.position = np.array(initial_pos, dtype=np.float64)

        if initial_velocity is None:
            self.velocity = np.array([8.0, 15.0], dtype=np.float64)
        else:
            self.velocity = np.array(initial_velocity, dtype=np.float64)

        self.radius = INITIAL_BALL_RADIUS

    def get_state(self):
        """
        Get current state of the ball.

        Returns:
            dict: Ball state with position, velocity, and radius
        """
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'radius': self.radius
        }


# Test the ball physics (run this file directly to see it work)
if __name__ == "__main__":
    print("Testing Ball Physics...")
    print("=" * 50)

    # Create a ball
    ball = Ball()
    print(f"Initial state: {ball.get_state()}")

    # Simulate a few timesteps
    print("\nSimulating 10 timesteps:")
    for i in range(10):
        ball.update()
        state = ball.get_state()
        print(f"Step {i+1}: pos={state['position']}, vel={state['velocity']}")

    # Test growing
    print("\nTesting ball growth:")
    print(f"Before: radius = {ball.radius}")
    ball.grow()
    print(f"After 1 growth: radius = {ball.radius}")
    ball.grow()
    print(f"After 2 growths: radius = {ball.radius}")

    print("\n" + "=" * 50)
    print("Ball physics test complete!")
