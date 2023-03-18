from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3.common.callbacks as clbks
from stable_baselines3 import DQN
import numpy as np
import os

# from stable_baselines3.common.callbacks import EvalCallback
from STBgym import ShutTheBoxEnv
from custom_logging import CustomTrackingCallback


wkdir = os.path.dirname(os.path.abspath(__file__))
tb_log_path = os.path.join(wkdir, "tb_logs/128_256_128_dqn_shut_the_box")
monitor_dir = os.path.join(wkdir, "monitor/")
save_path = os.path.join(wkdir, "final_model/128_256_128_dqn_shut_the_box")

# Create a callback for evaluation during training
callbacks = [
    clbks.StopTrainingOnMaxEpisodes(100_000_000, verbose=1),
]


env = make_vec_env(
    ShutTheBoxEnv,
    n_envs=20,
    monitor_dir=monitor_dir,
    # vec_env_cls=SubprocVecEnv,
    # vec_env_kwargs={"start_method": "fork"},
)
# env = Monitor(env, monitor_dir)
try:
    model = DQN.load(save_path, env=env)
except:
    model = DQN("MlpPolicy", env,learning_rate=1e-4,
            buffer_size=1_000_000,  # 1e6
            learning_starts=50_000,
            batch_size=1024,
            gamma=1.0,
            train_freq=64,
            gradient_steps=1,
            policy_kwargs={"net_arch": [128,256,128]},
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            target_update_interval=10_000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=0, 
            tensorboard_log=tb_log_path,
    )
total_reward = 0
done = False

model.learn(
    total_timesteps=100_000_000,
    callback=callbacks,
    log_interval=100,
    tb_log_name=tb_log_path,
    reset_num_timesteps=False
)

# num_eps = 10000
# rewards = []

# for ep in range(num_eps):
#     state = env.reset()
#     while not done:
#         action, _ = model.predict(state, deterministic=True)
#         state, reward, done, _ = env.step(action)
#         total_reward += reward
#         # print(action)
#     rewards.append(total_reward)
#     if ep % 1000 == 0:
#         print(f"avg reward: {np.mean(rewards[-100:])}")

# Save the trained model
model.save(save_path)
