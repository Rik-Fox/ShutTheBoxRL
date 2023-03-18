import os
import time
import numpy as np

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class CustomTrackingCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param log_dir: (str) Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, monitor_dir: str, verbose=0):
        super(CustomTrackingCallback, self).__init__(verbose)
        self.monitor_dir = monitor_dir

    def _init_callback(self) -> None:
        # Create folders if needed
        if self.monitor_dir is not None:
            os.makedirs(self.monitor_dir, exist_ok=True)

    def _on_rollout_end(self) -> None:
        pass

    def _on_step(self) -> bool:
        # Retrieve training reward
        x, y = ts2xy(load_results(self.monitor_dir), "timesteps")
        # this if basically stops from executing on 0th call
        if len(x) > 0:
            # Mean training reward over the last 100 episodes
            mean_reward = np.mean(y[-100:])
            self.logger.record("custom/mean_ep_rwd", mean_reward)
            # self.logger.record("custom/", )
            # self.logger.record("custom/", )
            # self.logger.record("custom/", )
            # self.logger.record("custom/", )
            # self.logger.record("custom/", )
            self.logger.dump(self.num_timesteps)
        if self.verbose > 0:
            print("--------------------------")

        return True

    def _on_training_end(self) -> None:
        self.model.save(os.path.join(self.monitor_dir, "final_model"), include="env")
