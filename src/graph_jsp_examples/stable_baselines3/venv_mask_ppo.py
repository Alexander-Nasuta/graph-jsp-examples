import sb3_contrib

import numpy as np

from typing import Dict

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from graph_jsp_env.disjunctive_graph_logger import log


def venv_basic_ppo_example(env_kwargs: Dict, total_timesteps=1_000, n_envs: int = 4):
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
    venv_basic_ppo_example(env_kwargs={
        "jps_instance": jsp,
        "perform_left_shift_if_possible": True,
        "normalize_observation_space": True,
        "flat_observation_space": True
    })
