import random, time
import pygame as pg
import numpy as np


class Dice:
    def __init__(self, screen, num_dice=1, rolling=12) -> None:
        self.screen = screen  # screen to draw on
        self.num_dice = num_dice  # number of dice to draw

        self.dw = (self.screen.get_width() // 8) - (num_dice * 2)  # size of dice
        self.dh = self.screen.get_height() // 8  # size of dice

        size = self.dw if self.dw >= self.dh else self.dh

        # self.dh = self.screen.get_height() - self.dh

        self.size = size  # Size of window/dice
        self.spsz = size // 10  # size of spots
        self.m = int(size / 2)  # mid-point of dice (or die?)
        self.l = self.t = int(size / 4)  # location of left and top spots
        self.r = self.b = self.size - self.l  # location of right and bottom spots

        self.rolling = rolling  # times that dice rolls before stopping
        self.diecol = (255, 255, 127)  # die colour
        self.spotcol = (0, 127, 127)  # spot colour
        self.surf_array = []  # array of surfaces to draw on
        for die in range(num_dice):
            self.surf_array.append(
                pg.Surface((self.size, self.size))
            )  # surface for each die

    def draw_dice(self, n, die):
        die.fill(self.diecol)  # clear previous spots
        if n % 2 == 1:
            # middle spot
            pg.draw.circle(die, self.spotcol, (self.m, self.m), self.spsz)
        if n == 2 or n == 3 or n == 4 or n == 5 or n == 6:
            # left bottom
            pg.draw.circle(die, self.spotcol, (self.l, self.b), self.spsz)
            # right top
            pg.draw.circle(die, self.spotcol, (self.r, self.t), self.spsz)
        if n == 4 or n == 5 or n == 6:
            # left top
            pg.draw.circle(die, self.spotcol, (self.l, self.t), self.spsz)
            # right bottom
            pg.draw.circle(die, self.spotcol, (self.r, self.b), self.spsz)
        if n == 6:
            # middle bottom
            pg.draw.circle(die, self.spotcol, (self.m, self.b), self.spsz)
            # middle top
            pg.draw.circle(die, self.spotcol, (self.m, self.t), self.spsz)

    def roll(self):
        """
        Apdapted from:
        DiceSimulator.py
        Author: Alan Richmond, Python3.codes
        https://tuxar.uk/a-graphical-dice-simulator/
        """
        for r in range(self.rolling):  # roll the die…
            for i, die in enumerate(self.surf_array):
                n = random.randint(1, 6)  # random number between 1 &amp;amp; 6
                self.draw_dice(n, die)
            self.draw()

    def update(self, dice=(None)):
        self.roll()
        for i, value in enumerate(dice):
            die = self.surf_array[i]
            self.draw_dice(value, die)
        self.draw()

    def draw(self):
        for i, die in enumerate(self.surf_array):
            if self.num_dice == 1:
                self.screen.blit(
                    die,
                    (
                        ((self.screen.get_width() / 2) - self.dw / 2),
                        self.screen.get_height() - (self.dh * (5 / 2)),
                    ),
                ),

            else:
                if i == 0:
                    self.screen.blit(
                        die,
                        (
                            ((self.screen.get_width() / 2) - self.dw * 2),
                            self.screen.get_height() - (self.dh * (5 / 2)),
                        ),
                    )
                else:
                    self.screen.blit(
                        die,
                        (
                            ((self.screen.get_width() / 2) + self.dw),
                            self.screen.get_height() - (self.dh * (5 / 2)),
                        ),
                    )

        pg.display.flip()
        time.sleep(0.2)


if __name__ == "__main__":
    screen = pg.display.set_mode((640, 480))
    pg.display.set_caption("Dice Simulator")

    d = Dice(screen, num_dice=2)

    for i in range(4):
        d.roll()
        pg.display.flip()
        time.sleep(0.2)
