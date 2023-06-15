import gym
import os
import time
import wandb as wb

import pathlib as pl

import sb3_contrib

from statistics import mean
from typing import List
from types import ModuleType

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from graph_jsp_env.disjunctive_graph_logger import log

import graph_jsp_utils.jsp_instance_downloader as jsp_downloader

from graph_jsp_examples.instance_loader import get_instance_by_name_as_numpy_array

gym.envs.register(
    id='gjsp-v0',
    entry_point='graph_jsp_env.disjunctive_graph_jsp_env:DisjunctiveGraphJspEnv',
    kwargs={},
)


class GraphJspLoggerCallback(BaseCallback):

    def __init__(self, optimal_makespan: float, wandb_ref: ModuleType = wb, verbose=0):
        super(GraphJspLoggerCallback, self).__init__(verbose)

        self.wandb_ref = wandb_ref
        self.start_time = None

        self.optimal_makespan = optimal_makespan

        self.total_left_shifts = 0

        self.senv_fields = [
            "makespan",
        ]
        self.venv_fields = [
        ]

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()

    def _get_vals(self, field: str) -> List:
        return [env_info[field] for env_info in self.locals['infos'] if field in env_info.keys()]

    def _on_step(self) -> bool:
        elapsed_time = time.perf_counter() - self.start_time
        logs = {
            "num_timesteps": self.num_timesteps,
            "time [sec]": elapsed_time,
            "time [min]": elapsed_time / 60,
        }
        ls_list = self._get_vals("left_shift")
        if len(ls_list):
            self.total_left_shifts += sum(ls_list)
        if self.wandb_ref:
            logs = {
                **{
                    f"{f}_env_{i}": info[f]
                    for i, info in enumerate(self.locals['infos'])
                    for f in self.senv_fields
                    if f in info.keys()
                },
                **{
                    f"optimality_gap_env_{i}": info["makespan"] / self.optimal_makespan - 1.0
                    for i, info in enumerate(self.locals['infos'])
                    if "makespan" in info.keys()
                },
                **{f"{f}_mean": mean(self._get_vals(f)) for f in self.senv_fields if self._get_vals(f)},
                **{
                    f"optimality_gap": mean(self._get_vals("makespan")) / self.optimal_makespan - 1.0 for _ in [1]
                    if self._get_vals("makespan")
                },
                **{f"{f}": mean(self._get_vals(f)) for f in self.venv_fields if self._get_vals(f)},
                "total_left_shifts": self.total_left_shifts,
                "left_shift_pct": self.total_left_shifts / self.num_timesteps * 100,

                "reward_mean": mean(self.locals["rewards"]),
                # reward_mean_{i} was a typo. Should be reward_{i}. keeping it for compatibility
                **{f"reward_mean_{i}": rew for i, rew in enumerate(self.locals['rewards'])},
                **{f"reward_{i}": rew for i, rew in enumerate(self.locals['rewards'])},

                **logs
            }
            self.wandb_ref.log(logs)

        return True


if __name__ == '__main__':
    log.info("downloading jsp instance.")
    target_dir = pl.Path(os.path.abspath(__file__)) \
        .parent \
        .parent \
        .parent \
        .joinpath("resources") \
        .joinpath("jsp_instances")

    jsp_downloader.download_instances(
        target_directory=target_dir,
        start_id=6,
        end_id=6
    )

    jsp = get_instance_by_name_as_numpy_array("ft06")


    def mask_fn(env):
        return env.valid_action_mask()


    venv = make_vec_env(
        env_id='gjsp-v0',
        env_kwargs={
            "jps_instance": jsp,

            "normalize_observation_space": True,
            "flat_observation_space": True,
            "perform_left_shift_if_possible": True,
            "reward_function": 'nasuta',
            "reward_function_parameters": {
                "scaling_divisor": 1.0
            },
            "default_visualisations": [
                "gantt_window",
            ]
        },
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        n_envs=8
    )

    venv.reset()
    model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env=venv,
        verbose=1,
        device='auto'  # cpu, mps (mac), cuda
    )

    log.info("training...")

    with wb.init(
            sync_tensorboard=False,
            monitor_gym=False,  # auto-upload videos, imgs, files etc.
            save_code=False,    # optional
            project="dev",      # specify your project here
    ) as run:
        logger_cb = GraphJspLoggerCallback(
            optimal_makespan=55.0,
            wandb_ref=wb
        )
        wb_cb = WandbCallback(
            gradient_save_freq=100,
            verbose=1,
        )
        model.learn(total_timesteps=1_000, progress_bar=False, callback=[wb_cb, logger_cb])
