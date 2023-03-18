import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import stable_baselines3.common.callbacks as clbks
import os

# from stable_baselines3.common.callbacks import EvalCallback
from STBgym import ShutTheBoxEnv
from custom_logging import CustomTrackingCallback


wkdir = os.path.dirname(os.path.abspath(__file__))
tb_log_path = os.path.join(wkdir, "tb_logs/")
monitor_dir = os.path.join(wkdir, "monitor/")
save_path = os.path.join(wkdir, "final_model/dqn_shut_the_box")


# def make_env(rank, seed=0):
#     def _init():
#         env = ShutTheBoxEnv()
#         env.seed(seed + rank)
#         return env

#     return _init


# # Set the number of parallel environments
# num_envs = 4
# env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method="fork")

env = make_vec_env(
    ShutTheBoxEnv,
    n_envs=6,
    monitor_dir=monitor_dir,
    # vec_env_cls=SubprocVecEnv,
    # vec_env_kwargs={"start_method": "fork"},
)

# Create the evaluation environment
eval_env = make_vec_env(
    ShutTheBoxEnv,
    n_envs=1,
    monitor_dir=monitor_dir,
)


# Create a callback for evaluation during training
callbacks = [
    clbks.EveryNTimesteps(
        1000,
        CustomTrackingCallback(
            monitor_dir=monitor_dir,
        ),
    ),
    clbks.EvalCallback(
        eval_env, best_model_save_path="./models/", log_path="./logs/", eval_freq=500
    ),
    clbks.StopTrainingOnMaxEpisodes(100000, verbose=1),
]

model = DQN("MlpPolicy", env,
        learning_rate=1e-4,
        buffer_size=1_000_000,  # 1e6
        learning_starts=50000,
        batch_size=1024,
        gamma=1.0,
        train_freq=64,
        gradient_steps=1,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        target_update_interval=10000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1, 
        tensorboard_log=tb_log_path
)
# Train the agent
model.learn(
    total_timesteps=1000000,
    callback=callbacks,
    log_interval=100,
    tb_log_name=tb_log_path,
)

# Save the trained model
model.save(save_path)
