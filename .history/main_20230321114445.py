from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3.common.callbacks as clbks
from stable_baselines3.common import utils
from stable_baselines3 import DQN
import numpy as np
import os

# from stable_baselines3.common.callbacks import EvalCallback
from STBgym import ShutTheBoxEnv
from custom_logging import CustomTrackingCallback


wkdir = os.path.dirname(os.path.abspath(__file__))
tb_log_path = os.path.join(wkdir, "tb_logs/128_256_128_dqn_shut_the_box")
cp_log_path = os.path.join(wkdir, "checkpoints/")
monitor_dir = os.path.join(wkdir, "monitor/")
save_path = os.path.join(wkdir, "final_model/128_256_128_dqn_shut_the_box")

# Create a callback for evaluation during training
callbacks = [
    clbks.CheckpointCallback(
        10_000,
        cp_log_path,
        name_prefix="128_256_128_dqn_shut_the_box_at",
        verbose=1,
    ),
    clbks.StopTrainingOnMaxEpisodes(1_000_000_000, verbose=1),
]


env = make_vec_env(
    ShutTheBoxEnv,
    n_envs=200,
    monitor_dir=monitor_dir,
    # vec_env_cls=SubprocVecEnv,
    # vec_env_kwargs={"start_method": "fork"},
)
# env = Monitor(env, monitor_dir)
try:
    model = DQN.load(save_path, env=env)
    model.verbose = 0
    model._episode_num = model._episode_num
    model._n_calls = model._n_calls
    model._n_updates = model._n_updates
    model.num_timesteps = model.num_timesteps
    model.replay_buffer = model.replay_buffer
    model.learning_starts = 0
    # model.load_replay_buffer(model.replay_buffer)
    model.exploration_schedule = utils.get_linear_fn(
        model.exploration_final_eps,
        model.exploration_final_eps,
        0,
    )
except:
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=1_000_000,  # 1e6
        learning_starts=50_000,
        batch_size=1024,
        gamma=1.0,
        train_freq=64,
        gradient_steps=1,
        policy_kwargs={"net_arch": [128, 256, 128]},
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        target_update_interval=10_000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=tb_log_path,
    )

model.learn(
    total_timesteps=1_000_000_000,
    callback=callbacks,
    log_interval=100,
    tb_log_name=tb_log_path,
    reset_num_timesteps=False,
)

model.save(
    save_path,
)
