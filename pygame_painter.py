import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Loading pre-trained model
model = load_model("model.h5")

# Initializing Pygame
pygame.init()

# Initializing display
WIDTH, HEIGHT = 560, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw Digit and Predict")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

clear_button_rect = pygame.Rect(170, 570, 100, 30)
clear_button_text = "Clear"

digit_display_rect = pygame.Rect(290, 570, 100, 30)
predicted_digit = ''

canvas_surface = pygame.Surface((560, 560))
canvas_surface.fill(WHITE)

def draw_buttons():
    font = pygame.font.SysFont(None, 24)

    pygame.draw.rect(screen, GRAY, clear_button_rect)
    clear_text_surface = font.render(clear_button_text, True, BLACK)
    clear_text_rect = clear_text_surface.get_rect(center=clear_button_rect.center)
    screen.blit(clear_text_surface, clear_text_rect)

    font = pygame.font.SysFont(None, 40)
    pygame.draw.rect(screen, GRAY, digit_display_rect)
    predict_text_surface = font.render(str(predicted_digit), True, BLACK)
    predict_text_rect = predict_text_surface.get_rect(center=digit_display_rect.center)
    screen.blit(predict_text_surface, predict_text_rect)
    pygame.draw.rect(screen, BLACK, digit_display_rect, 2)


def get_canvas_array(surface):
    small_surface = pygame.transform.smoothscale(surface, (28, 28))
    
    pixel_array = pygame.surfarray.array3d(small_surface)
    pixel_array = np.transpose(pixel_array, (1, 0, 2))

    gray = pixel_array.mean(axis=2)
    
    gray = (255 - gray) / 255.0
    
    return gray.astype(np.float32)

def predict_digit(input_img):
    prediction = model.predict(input_img, verbose=0)
    return np.argmax(prediction)

running = True
while running:
    screen.fill(WHITE)
    screen.blit(canvas_surface, (0, 0))
    draw_buttons()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos

            if clear_button_rect.collidepoint(mouse_pos):
                canvas_surface.fill(WHITE)

        elif event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                pygame.draw.circle(canvas_surface, BLACK, pos, 15)
                input_img = get_canvas_array(canvas_surface).reshape(1, 28, 28, 1)
                predicted_digit = predict_digit(input_img)
                
                pygame.draw.rect(screen, (200, 200, 200), digit_display_rect)
                pygame.draw.rect(screen, (0, 0, 0), digit_display_rect, 2)

                font = pygame.font.SysFont(None, 40)
                digit_surface = font.render(str(predicted_digit), True, (0, 0, 0))

                digit_rect = digit_surface.get_rect(center=digit_display_rect.center)
                screen.blit(digit_surface, digit_rect)

    pygame.display.flip()
pygame.quit()
