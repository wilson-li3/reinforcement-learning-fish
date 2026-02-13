import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-400, 400)
ax.set_ylim(-400, 400)
ax.set_aspect('equal')
ax.set_facecolor('black')

# Draw circle boundary
circle = plt.Circle((0, 0), 300, fill=False, color='white', linewidth=2)
ax.add_patch(circle)

# Ball properties
ball_pos = np.array([0.0, -100.0])
ball_velocity = np.array([4.0, 3.0])
ball_radius = 15
circle_radius = 300

# Create ball
ball, = ax.plot([], [], 'ro', markersize=15)

def init():
    ball.set_data([], [])
    return ball,

def animate(frame):
    global ball_pos, ball_velocity

    # Update ball position
    ball_pos += ball_velocity

    # Calculate distance from ball to circle center
    distance = np.linalg.norm(ball_pos)

    # Check collision with circle boundary
    if distance + ball_radius >= circle_radius:
        # Normalize the direction vector from center to ball
        normal = ball_pos / distance

        # Calculate dot product of velocity and normal
        dot = np.dot(ball_velocity, normal)

        # Reflect velocity vector
        ball_velocity -= 2 * dot * normal

        # Push ball back inside the circle
        overlap = (distance + ball_radius) - circle_radius
        ball_pos -= overlap * normal

    # Update ball position on plot
    ball.set_data([ball_pos[0]], [ball_pos[1]])

    return ball,

# Create animation
ani = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=None, interval=20, blit=True, repeat=True)

plt.title('Bouncing Ball in Circle', color='white', fontsize=16)
plt.show()
