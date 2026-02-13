import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('black')
ax.set_xlim(-400, 400)
ax.set_ylim(-400, 400)
ax.set_aspect('equal')
ax.set_facecolor('black')
ax.axis('off')

# Draw circle boundary
circle = plt.Circle((0, 0), 300, fill=False, color='white', linewidth=2)
ax.add_patch(circle)

# Ball properties
ball_pos = np.array([0.0, -100.0])
ball_velocity = np.array([8.0, 15.0])  # Much faster initial velocity for higher bouncing
ball_radius = 15
circle_radius = 300
gravity = 0.3  # Gravity acceleration

# Create ball (white instead of red)
ball, = ax.plot([], [], 'wo', markersize=15)

# Position text display
pos_text = ax.text(0, 350, '', color='white', ha='center', fontsize=12,
                   family='monospace')

def init():
    ball.set_data([], [])
    pos_text.set_text('')
    return ball, pos_text

def animate(frame):
    global ball_pos, ball_velocity

    # Apply gravity
    ball_velocity[1] -= gravity

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

        # Reflect velocity vector (perfectly elastic collision)
        ball_velocity -= 2 * dot * normal

        # Push ball back inside the circle
        overlap = (distance + ball_radius) - circle_radius
        ball_pos -= overlap * normal

    # Update ball position on plot
    ball.set_data([ball_pos[0]], [ball_pos[1]])

    # Update position text
    pos_text.set_text(f'Position: ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})')

    return ball, pos_text

# Create animation
ani = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=None, interval=16, blit=True, repeat=True,
                             cache_frame_data=False)

plt.show()
