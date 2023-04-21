import random, time
import pygame as pg
import numpy as np


class dice:
    def __init__(self, screen, num_dice=1, rolling=12) -> None:
        self.screen = screen  # screen to draw on
        self.num_dice = num_dice  # number of dice to draw

        self.dw = ((self.screen.get_width() // num_dice) // 4) - (
            num_dice * 2
        )  # size of dice
        self.dh = self.screen.get_height() // (num_dice - 1) // 4  # size of dice

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
                die.fill(self.diecol)  # clear previous spots
                if n % 2 == 1:
                    pg.draw.circle(
                        die, self.spotcol, (self.m, self.m), self.spsz
                    )  # middle spot
                if n == 2 or n == 3 or n == 4 or n == 5 or n == 6:
                    pg.draw.circle(
                        die, self.spotcol, (self.l, self.b), self.spsz
                    )  # left bottom
                    pg.draw.circle(
                        die, self.spotcol, (self.r, self.t), self.spsz
                    )  # right top
                if n == 4 or n == 5 or n == 6:
                    pg.draw.circle(
                        die, self.spotcol, (self.l, self.t), self.spsz
                    )  # left top
                    pg.draw.circle(
                        die, self.spotcol, (self.r, self.b), self.spsz
                    )  # right bottom
                if n == 6:
                    pg.draw.circle(
                        die, self.spotcol, (self.m, self.b), self.spsz
                    )  # middle bottom
                    pg.draw.circle(
                        die, self.spotcol, (self.m, self.t), self.spsz
                    )  # middle top

                if i % 2 == 0:
                    self.screen.blit(
                        die,
                        (
                            (
                                (self.screen.get_width() / 2)
                                - (self.dw * np.clip((i / 2) - 1, 0, self.num_dice))
                            ),
                            self.screen.get_height() - (self.dh * 1.5),
                        ),
                    )
                else:
                    self.screen.blit(
                        die,
                        (
                            (
                                (self.screen.get_width() / 2)
                                + (
                                    self.dw
                                    * np.clip(np.ceil(i / 2) - 1, 0, self.num_dice)
                                )
                            ),
                            self.screen.get_height() - (self.dh * 1.5),
                        ),
                    )

            pg.display.flip()
            time.sleep(0.2)


if __name__ == "__main__":
    screen = pg.display.set_mode((640, 480))
    pg.display.set_caption("Dice Simulator")

    d = dice(screen, num_dice=2)

    for i in range(4):
        d.roll()
        pg.display.flip()
        time.sleep(0.2)
