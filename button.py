import pygame


class Button:
    def __init__(self, text, font, color, rect):
        self.text = text
        self.font = font
        self.color = color
        self.rect = rect

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def contains_point(self, point):
        return self.rect.collidepoint(point)
