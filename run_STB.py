import gym
from stable_baselines3 import DQN
import numpy as np

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from maskedDQN import MaskableDQNPolicy, MaskableDQN
from STBgym import ShutTheBoxEnv


# Load the trained model
model = MaskableDQN.load(
    "/home/rfox/ShutTheBoxRL/final_model/1_maskableDQN_stb_64_128_64.zip"
)


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_masks()


# Test the trained agent
env = ShutTheBoxEnv()
env = ActionMasker(env, mask_fn)  # Wrap to enable masking


rewards = []
boxShuts = 0
num_eps = 1_000_000
done = False

for ep in range(num_eps):
    state = env.reset()
    env.seed(ep)
    done = False

    while not done:
        action_masks = get_action_masks(env)
        if action_masks.any():
            action, _ = model.predict(state, action_masks=action_masks)
        else:
            # No actions available, choose a random action to let episode end
            action = np.asarray(env.action_space.sample())

        state, reward, done, _ = env.step(action.item())
        # if ep % 10_000 == 0:
        #     print(env.numbers, "  ", env.dice_sum)
        #     print(env.actions[action])
        # print(reward)
        rewards.append(reward)
        if reward == 10:
            boxShuts += 1

    if ep % 10_000 == 0:
        print(
            f"Avg reward at episode {ep}: {np.sum(rewards[-100:])/100}    SD: {np.std(rewards[-100:])}"
        )
        # print(rewards[-10:])
        # print(env.numbers)
print("Sucess rate : ", boxShuts / num_eps)
