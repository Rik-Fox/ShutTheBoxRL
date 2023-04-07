import argparse
import numpy as np
import os
import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3.common.callbacks as clbks
from stable_baselines3.common import utils

from sb3_contrib.common.wrappers import ActionMasker
from maskedDQN import MaskableDQNPolicy, MaskableDQN

from STBgym import ShutTheBoxEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_masks()


def Main(args):
    wkdir = os.path.dirname(os.path.abspath(__file__))
    # build the architecture string
    arch_str = ""
    for layer in range(len(args.net_arch)):
        arch_str += f"_{args.net_arch[layer]}"
    # remove the first underscore
    arch_str = arch_str[1:]

    # set up the paths for the checkpoints, monitor and tensorboard logs
    cp_log_path = os.path.join(wkdir, "checkpoints", f"{args.model_name}_{arch_str}/")
    monitor_dir = os.path.join(wkdir, "monitor", f"{args.model_name}_{arch_str}")

    # Create the environment
    env = make_vec_env(
        ShutTheBoxEnv,
        50,
        env_kwargs=None,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        seed=1234,
        monitor_dir=monitor_dir,
    )
    # env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    env.reset()

    # Create a callback for checkpoints during training
    callbacks = [
        clbks.CheckpointCallback(
            1_000,
            cp_log_path,
            name_prefix=f"model_at",
            verbose=1,
        ),
        clbks.StopTrainingOnMaxEpisodes(1_000_000, verbose=1),
    ]

    # Load the model if it exists
    try:
        m_s = os.listdir(os.path.join(wkdir, "final_model"))
        latest_m = np.max([int(m.split("_")[0]) for m in m_s])

        load_path = os.path.join(
            wkdir, "final_model", f"{latest_m}_{args.model_name}_{arch_str}"
        )
        save_path = os.path.join(
            wkdir, "final_model", f"{latest_m+1}_{args.model_name}_{arch_str}"
        )

        tb_log_path = os.path.join(
            wkdir, "tb_logs", f"{latest_m+1}_{args.model_name}_{arch_str}"
        )

        model = MaskableDQN.load(load_path, env=env)
        model.verbose = args.verbose
        model.tensorboard_log = tb_log_path

        # not sure if actually needed but I have
        # had to do this before so better safe than sorry
        model._episode_num = model._episode_num
        model._n_calls = model._n_calls  # number of calls to learn()
        model._n_updates = model._n_updates
        model.num_timesteps = model.num_timesteps
        model.replay_buffer = model.replay_buffer
        model.learning_starts = 0  # start learning immediately
        # model.load_replay_buffer(model.replay_buffer)

        # this is set so only explores when first created
        # TODO: make these argparse args so is adaptable
        model.exploration_schedule = utils.get_linear_fn(
            model.exploration_final_eps,
            model.exploration_final_eps,
            0,
        )
    # if the model does not exist, create a new one
    except FileNotFoundError:
        # save path for the final model
        save_path = os.path.join(
            wkdir, "final_model", f"{0}_{args.model_name}_{arch_str}"
        )
        tb_log_path = os.path.join(
            wkdir, "tb_logs", f"{0}_{args.model_name}_{arch_str}"
        )

        model = MaskableDQN(
            MaskableDQNPolicy,
            env,
            learning_rate=1e-4,
            buffer_size=1_000_000,  #  replay buffer size
            learning_starts=50_000,  # number of steps before learning starts
            batch_size=1024,
            tau=1.0,  # target network update rate (1 for hard update)
            gamma=0.99,
            train_freq=64,
            gradient_steps=1,
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            target_update_interval=10_000,  # update the target network every 10_000 steps
            exploration_fraction=0.01,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log=tb_log_path,
            policy_kwargs={"net_arch": args.net_arch},  # NN architecture
            verbose=args.verbose,
            seed=1234,
            device="auto",
            _init_setup_model=True,
        )

    model.learn(
        total_timesteps=9_000_000,
        callback=callbacks,
        log_interval=100,
        tb_log_name=tb_log_path,
        reset_num_timesteps=False,
    )

    model.save(
        save_path,
    )


if __name__ == "__main__":
    param_parser = argparse.ArgumentParser(description="ShutTheBoxRL")
    param_parser.add_argument(
        "--model_name",
        type=str,
        default="maskableDQN_stb",
    )
    param_parser.add_argument(
        "--net_arch",
        nargs="+",
        type=list,
        default=[64, 128, 64],
        help="List of NN layers to be built for a model",
    )
    param_parser.add_argument(
        "--verbose",
        type=int,
        default=0,
    )

    args = param_parser.parse_args()

    Main(args)
