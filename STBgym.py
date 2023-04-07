import gym
from gym import spaces
import itertools
import random
import numpy as np


class ShutTheBoxEnv(gym.Env):
    def __init__(self, seed=None):
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
        self.dice_sum = None

    def seed(self, seed=None):
        random.seed(seed)
        self.action_space.seed(seed)

    def _get_state(self):
        return [int(number in self.numbers) for number in range(1, 10)]

    def reset(self):
        self.numbers = list(range(1, 10))
        self.dice_sum = sum(self.roll_dice())
        state = self._get_state()
        state.append(self.dice_sum / 12)
        # print(state)
        return state

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
                self.dice_sum = sum(self.roll_dice())
            else:
                reward = -100  # heavy penalty for invalid moves
                done = True

            if not bool(self.numbers):  # game is done if there are no numbers left
                done = True
                reward = 10

        state = self._get_state()
        state.append(self.dice_sum / 12)
        # print(state)

        return state, reward, done, {}

    def get_valid_moves(self, dice_sum):
        valid_moves = []
        for r in range(1, len(self.numbers) + 1):
            for combination in itertools.combinations(self.numbers, r):
                if sum(combination) == dice_sum:
                    valid_moves.append(combination)
        return valid_moves
    
    def action_masks(self):
        valid_moves = self.get_valid_moves(self.dice_sum)
        
        return [move in valid_moves for move in self.actions]
