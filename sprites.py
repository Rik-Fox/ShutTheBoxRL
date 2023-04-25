import pygame


class NumberSprites(pygame.sprite.Sprite):
    def __init__(self, font_size=64):
        super().__init__()
        self.font = pygame.font.SysFont("Arial", font_size)
        self.surface = {}
        for number in range(1, 10):
            number_str = str(number)
            self.surface[number_str] = self.font.render(
                number_str, True, (255, 255, 255)
            )

    def render(self, number, position, color):
        number_str = str(number)
        number_surface = self.surface[number_str]
        number_surface.set_colorkey((0, 0, 0))
        number_surface.fill(color, None, pygame.BLEND_RGB_MULT)
        rect = number_surface.get_rect()
        rect.center = position
        return number_surface, rect

    def update(self):
        self.render_number()


class ActionSprites(pygame.sprite.Sprite):
    def __init__(self, actions, font_size=64):
        super().__init__()
        self.font = pygame.font.SysFont("Arial", font_size)
        self.surface = {}
        for i, action in enumerate(actions):
            act_str = str(action)
            self.surface[act_str] = self.font.render(act_str, True, (255, 255, 255))

    def render(self, act, position, color):
        act_str = str(act)
        surface = self.surface[act_str]
        surface.set_colorkey((0, 0, 0))
        surface.fill(color, None, pygame.BLEND_RGB_MULT)
        rect = surface.get_rect()
        rect.center = position
        return surface, rect

    def update(self):
        self.render_number()
