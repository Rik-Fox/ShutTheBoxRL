import gym
from gym import spaces
import itertools
import sys
import random
import numpy as np
import pygame as pg
from dice import Dice
from sprites import NumberSprites, ActionSprites
from button import Button


class ShutTheBoxEnv(gym.Env):
    def __init__(self, seed=None, headless=True):
        super(ShutTheBoxEnv, self).__init__()
        self.numbers = list(range(1, 10))

        self.actions = []
        for size in range(1, 10):  # Loop over all possible combination sizes
            for comb in itertools.combinations(range(1, 10), size):
                if sum(comb) <= 12 and sum(comb) > 1:
                    self.actions.append(comb)

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.numbers) + 1,), dtype=np.float16
        )
        if seed is None:
            seed = np.random.randint(1000)
        self.seed(seed=seed)
        self.die1 = None
        self.die2 = None
        self.dice_sum = None
        self.headless = headless
        if not self.headless:
            self._setup_render()

    def seed(self, seed=None):
        random.seed(seed)
        self.action_space.seed(seed)

    def _get_state(self):
        return [int(number in self.numbers) for number in range(1, 10)]

    def reset(self):
        self.numbers = list(range(1, 10))
        self.die1, self.die2 = self.roll_dice()
        self.dice_sum = sum([self.die1, self.die2])
        state = self._get_state()
        state.append(self.dice_sum / 12)
        # print(state)
        return np.asarray(state)

    def roll_dice(self):
        if set([7, 8, 9]).issubset(set(self._get_state())):
            return (random.randint(1, 6),)  # Roll one die
        else:
            return (random.randint(1, 6), random.randint(1, 6))  # Roll two dice

    def step(self, action):
        valid_moves = self.get_valid_moves(self.dice_sum)
        # print(action)
        done = False

        if not valid_moves:
            reward = -sum(
                self.numbers
            )  # negative reward, based on the sum of the remaining numbers
            done = True
        else:
            move = self.actions[action]
            # print(move)
            # print(self.dice_sum)

            if move in set(valid_moves):  # and sum(move) == self.dice_sum:
                for num in move:
                    self.numbers.remove(num)
                reward = 0
                # only get rolled once found a valid move
                self.die1, self.die2 = self.roll_dice()
                self.dice_sum = sum([self.die1, self.die2])
            else:
                reward = -100  # heavy penalty for invalid moves
                done = True

            if not bool(self.numbers):  # game is done if there are no numbers left
                done = True
                reward = 10

        state = self._get_state()
        state.append(self.dice_sum / 12)
        # print(state)

        return np.asarray(state), reward, done, {}

    def get_valid_moves(self, dice_sum):
        valid_moves = []
        for r in range(1, len(self.numbers) + 1):
            for combination in itertools.combinations(self.numbers, r):
                if sum(combination) == dice_sum:
                    valid_moves.append(combination)
        return valid_moves

    def action_masks(self):
        valid_moves = self.get_valid_moves(self.dice_sum)

        return np.asarray([move in valid_moves for move in self.actions])

    def _setup_render(
        self,
    ):
        WINDOW_WIDTH = 1280
        WINDOW_HEIGHT = 960
        pg.init()
        self.screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pg.display.set_caption("Shut The Box")
        self.running = True
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Arial", 30)

        self.numberSprites = NumberSprites(font_size=164)
        self.actionSprites = ActionSprites(self.actions, font_size=64)

        # Define the end game button
        button_text = "End Game"
        button_width = 150
        button_height = 65
        button_x = WINDOW_WIDTH - button_width - 20
        button_y = WINDOW_HEIGHT - button_height - 20
        button_rect = pg.Rect(button_x, button_y, button_width, button_height)
        self.endButton = Button(button_text, self.font, (255, 0, 0), button_rect)

        # Define the end game button
        button_text = "Conifrm Move"
        button_width = 150
        button_height = 65
        button_x = 20
        button_y = WINDOW_HEIGHT - button_height - 20
        button_rect = pg.Rect(button_x, button_y, button_width, button_height)
        self.selectMoveButton = Button(button_text, self.font, (255, 0, 0), button_rect)

    def render(self, action=None, mode="eval"):
        if self.headless:
            raise ("Cannot render in headless mode")

        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)

        if action:
            rolled = True
        else:
            rolled = False

        if set([7, 8, 9]).issubset(set(self._get_state())):
            self.dice = Dice(self.screen, num_dice=1, rolling=12)
        else:
            self.dice = Dice(self.screen, num_dice=2, rolling=12)

        while self.running:
            # Handle events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                    self.done = True
                    self.close()
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if self.endButton.contains_point(event.pos):
                        self.running = False
                        self.done = True
                        self.close()

            # Clear the screen
            self.screen.fill(BLACK)

            # draw the end game button
            self.endButton.draw(self.screen)
            # Draw the numbers
            for i in range(1, 10):
                if i in self.numbers:
                    colour = WHITE
                else:
                    colour = RED

                if (action is not None) and (i in self.actions[action]):
                    colour = GREEN

                number, rect = self.numberSprites.render(i, (120 * i, 100), colour)
                self.screen.blit(number, rect)

            # show the action
            if action is not None:
                text = self.font.render(
                    "Action: " + str(self.actions[action]), True, GREEN
                )
                self.screen.blit(text, (20, 60))

            # render valid moves
            valid_moves = self.get_valid_moves(self.dice_sum)
            i = 0
            rect = pg.Rect(20, 300, 40, 40 * len(valid_moves))
            for ac in self.actions:
                if ac in valid_moves:
                    if (action is not None) and (ac == self.actions[action]):
                        colour = GREEN
                    else:
                        colour = WHITE
                    act, rect = self.actionSprites.render(
                        ac, (rect.x + rect.width + 160, 300), colour
                    )
                    self.screen.blit(act, rect)
                    i += 1

            # text = self.font.render("Valid moves: ", True, WHITE)
            # self.screen.blit(text, (20, 100))
            # valid_moves = self.get_valid_moves(self.dice_sum)
            # text = self.font.render(str(valid_moves), True, WHITE)
            # self.screen.blit(text, (20, 140))

            # render dice
            if not rolled:
                self.dice.update((self.die1, self.die2))
                rolled = True
            self.dice.draw()

            # Update the display
            pg.display.flip()
            self.clock.tick(60)

            # Wait for a short time to prevent the game from running too fast
            pg.time.wait(1000)

            if mode == "human":
                # check which number surfaces have been clicked
                pass
                for event in pg.event.get():
                    if event.type == pg.MOUSEBUTTONDOWN:
                        for move in self.actionSprites:
                            if move.contains_point(event.pos):
                                self.clickedAction = move
                                self.selectMoveButton.draw(self.screen)
                        if self.clickedAction:
                            self.confirmMoveButton.draw(self.screen)
                        if self.confirmMoveButton.contains_point(event.pos):
                            self.numbers.remove(self.clickedAction.number)

            elif mode == "eval":
                pg.time.wait(2000)
                return self.running

    def close(self):
        # Clean up Pygame
        pg.quit()
        sys.exit()
