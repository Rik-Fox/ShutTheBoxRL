import gym
from stable_baselines3 import DQN
import numpy as np
from STBgym import ShutTheBoxEnv

# Load the trained model
model = DQN.load("final_model/dqn_shut_the_box")

# Test the trained agent
env = ShutTheBoxEnv()
state = env.reset()
total_reward = 0
done = False
while not done:
    action, _ = model.predict(state, deterministic=True)
    state, reward, done, _ = env.step(action)
    print(state)
    print(reward)
    if np.isnan(reward):
        raise ValueError("reward is NaN")
    total_reward += reward

print(f"Total reward: {total_reward}")

