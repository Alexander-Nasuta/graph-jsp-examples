import os
import sb3_contrib

import numpy as np
import pathlib as pl

from typing import Dict
from rich.progress import track
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env

from graph_jsp_env.disjunctive_graph_logger import log
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv


def venv_ppo_video_recorder_example(env_kwargs: Dict, total_timesteps=1_000, n_envs: int = 2):
    log.info("setting up vectorised environment")

    def mask_fn(env):
        return env.valid_action_mask()

    venv = make_vec_env(
        env_id=DisjunctiveGraphJspEnv,
        env_kwargs=env_kwargs,

        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},

        n_envs=n_envs)

    model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env=venv,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    log.info("setting up video recorder")
    episode_len = venv.envs[0].total_tasks_without_dummies

    venv = VecVideoRecorder(
        venv,
        pl.Path(os.path.abspath(__file__)).parent,
        record_video_trigger=lambda x: x == 0,
        video_length=episode_len,
        name_prefix=f"venv_mask_ppo_video")

    obs = venv.reset()
    for _ in track(range(episode_len), description="recording frames ..."):
        masks = np.array([env.action_masks() for env in model.env.envs])
        action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
        obs, _, _, _ = venv.step(action)

    # Save the video
    log.info("saving video...")
    venv.close()

    # somehow VecVideoRecorder crashes at the end of the script (when __del__() in VecVideoRecorder is called)
    # for some reason there are no issues when deleting env manually
    del venv
    log.info("done.")


if __name__ == '__main__':
    jsp = np.array([
        [
            [1, 2, 0],  # job 0
            [0, 2, 1]  # job 1
        ],
        [
            [17, 12, 19],  # task durations of job 0
            [8, 6, 2]  # task durations of job 1
        ]

    ])
    venv_ppo_video_recorder_example(env_kwargs={
        "jps_instance": jsp,
        "perform_left_shift_if_possible": True,
        "normalize_observation_space": True,
        "flat_observation_space": True,
        "default_visualisations": [
            "gantt_window",
            "graph_window",  # very expensive
        ],
    })
