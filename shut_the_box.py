import pygame
import sys
import random
import time

# Initialize pygame
pygame.init()

# Set screen size and title
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Shut the Box")

# Load font and images
font = pygame.font.Font(None, 36)
dice_images = [pygame.transform.scale(pygame.image.load(f'dice_images/dice_{i}.png'), (64, 64)) for i in range(1, 7)]

# Set colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def draw_numbers(numbers, selected, dice_roll):
    for i, number in enumerate(numbers):
        if number:
            color = GREEN if valid_move(selected, dice_roll) else RED
            color = color if selected[i] else BLACK
            text = font.render(str(i + 1), True, color)
            screen.blit(text, (50 + i * 50, 250))
    
    valid_moves = list_valid_moves(numbers, dice_roll)
    #text = font.render(f"Valid Moves: {valid_moves}", True, BLACK)
    #screen.blit(text, (50, 350))

def draw_dice(roll):
    for i, r in enumerate(roll):
        screen.blit(dice_images[r - 1], (50 + i * 100, 100))

def roll_dice():
    return [random.randint(1, 6), random.randint(1, 6)]

def draw_confirm_button():
    pygame.draw.rect(screen, GRAY, (50, 300, 100, 50))
    text = font.render("Confirm", True, BLACK)
    screen.blit(text, (55, 310))

def valid_move(selected, dice_roll):
    selected_sum = sum(i + 1 for i, s in enumerate(selected) if s)
    return selected_sum == sum(dice_roll)

def no_valid_moves(numbers, dice_roll):
    valid_moves = list_valid_moves(numbers, dice_roll)
    return len(valid_moves) == 0

def list_valid_moves(numbers, dice_roll):
    valid_moves = []
    for i in range(1, 2**len(numbers)):
        binary = format(i, f'0{len(numbers)}b')
        combination = [int(b) * (i+1) for i, b in enumerate(binary) if int(b) and numbers[i]]
        if sum(combination) == sum(dice_roll) and combination not in valid_moves:
            valid_moves.append(combination)
    return valid_moves

def draw_reset_button():
    pygame.draw.rect(screen, BLUE, (50, 400, 100, 50))
    text = font.render("Reset", True, WHITE)
    screen.blit(text, (60, 410))

def reset_game():
    return [True] * 9, roll_dice(), [False] * 9, False

def draw_previous_scores(scores):
    text = font.render("Previous Scores", True, BLACK)
    screen.blit(text, (400, 50))
    for i, score in enumerate(scores):
        text = font.render(str(score), True, BLACK)
        screen.blit(text, (400, 100 + i * 30))

def display_valid_moves(valid_moves):
    text = font.render("Valid Moves:", True, BLACK)
    screen.blit(text, (400, 300))
    for i, move in enumerate(valid_moves):
        text = font.render(str(move), True, BLACK)
        screen.blit(text, (400 + (i % 2) * 150, 330 + (i // 2) * 30))


def game_over_message():
    text = font.render("Game Over", True, RED)
    screen.blit(text, (50, 360))

def no_valid_move_message():
    text = font.render("Game Over, No Valid Moves Remain", True, RED)
    screen.blit(text, (50, 400))

def handle_events(numbers, dice_roll, selected, game_over, scores):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()

            if 250 <= y <= 286 and not game_over:
                index = (x - 50) // 50
                if 0 <= index < 9 and numbers[index]:
                    selected[index] = not selected[index]

            elif 50 <= x <= 150 and 300 <= y <= 350 and not game_over:
                if valid_move(selected, dice_roll):
                    numbers = [n and not s for n, s in zip(numbers, selected)]
                    selected = [False] * 9
                    dice_roll = roll_dice()

            elif 50 <= x <= 150 and 400 <= y <= 450 and game_over:
                scores.append(45 - sum(i+1 for i, n in enumerate(numbers) if not n))
                numbers, dice_roll, selected, game_over = reset_game()

    return numbers, dice_roll, selected, game_over, scores
def game_loop():
    numbers = [True] * 9
    dice_roll = roll_dice()
    selected = [False] * 9
    game_over = False
    scores = []

    while True:
        # Handle events
        numbers, dice_roll, selected, game_over, scores = handle_events(numbers, dice_roll, selected, game_over, scores)

        # Clear the screen
        screen.fill(WHITE)

        # Draw numbers, dice roll, and confirm button
        draw_numbers(numbers, selected, dice_roll)
        draw_dice(dice_roll)
        draw_confirm_button()

        # Check for game over and display a message
        game_over = no_valid_moves(numbers, dice_roll)
        if game_over:
            game_over_message()
            draw_reset_button()

        # Draw previous scores
        draw_previous_scores(scores)

        # Display valid moves
        valid_moves = list_valid_moves(numbers, dice_roll)
        display_valid_moves(valid_moves)

        # Update display
        pygame.display.flip()

if __name__ == "__main__":
    game_loop()

