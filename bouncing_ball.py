import pygame
import math

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bouncing Ball in Circle")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (100, 150, 255)

# Circle properties
circle_center = (WIDTH // 2, HEIGHT // 2)
circle_radius = 300

# Ball properties
ball_pos = [WIDTH // 2, HEIGHT // 2 - 100]
ball_radius = 15
ball_velocity = [4, 3]

# Clock for controlling frame rate
clock = pygame.time.Clock()
FPS = 60

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update ball position
    ball_pos[0] += ball_velocity[0]
    ball_pos[1] += ball_velocity[1]

    # Calculate distance from ball to circle center
    dx = ball_pos[0] - circle_center[0]
    dy = ball_pos[1] - circle_center[1]
    distance = math.sqrt(dx**2 + dy**2)

    # Check collision with circle boundary
    if distance + ball_radius >= circle_radius:
        # Normalize the direction vector from center to ball
        nx = dx / distance
        ny = dy / distance

        # Calculate dot product of velocity and normal
        dot = ball_velocity[0] * nx + ball_velocity[1] * ny

        # Reflect velocity vector
        ball_velocity[0] -= 2 * dot * nx
        ball_velocity[1] -= 2 * dot * ny

        # Push ball back inside the circle
        overlap = (distance + ball_radius) - circle_radius
        ball_pos[0] -= overlap * nx
        ball_pos[1] -= overlap * ny

    # Clear screen
    screen.fill(BLACK)

    # Draw circle boundary
    pygame.draw.circle(screen, WHITE, circle_center, circle_radius, 3)

    # Draw ball
    pygame.draw.circle(screen, RED, (int(ball_pos[0]), int(ball_pos[1])), ball_radius)

    # Update display
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
