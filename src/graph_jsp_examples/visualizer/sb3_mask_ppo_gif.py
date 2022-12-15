import gym
import os
import imageio
import sb3_contrib

import numpy as np
import pathlib as pl
import stable_baselines3 as sb3

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from graph_jsp_env.disjunctive_graph_logger import log

from rich.progress import track
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


def train_and_create_gif(jsp_instance: np.ndarray,
                         total_timesteps: int = 2_000, filename: str = "mask_ppo_gif") -> None:
    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp_instance,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=True,
        action_mode='task',  # alternative 'job'
        dtype='float32'
    )
    env = sb3.common.monitor.Monitor(env)

    def mask_fn(env: gym.Env) -> np.ndarray:
        return env.valid_action_mask()

    env = ActionMasker(env, mask_fn)

    model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1
    )

    # Train the agent
    log.info("training the model")
    model.learn(total_timesteps=total_timesteps)

    images = []
    obs = model.env.reset()
    img = model.env.render(mode='rgb_array')
    images.append(img)
    for _ in track(range(model.env.envs[0].total_tasks_without_dummies), description="creating gif..."):
        action, _ = model.predict(
            obs,
            action_masks=model.env.envs[0].action_masks(),
            deterministic=False
        )
        obs, _, done, _ = model.env.step(action)
        img = model.env.render(mode='rgb_array')
        images.append(img)

    imageio.mimsave(
        pl.Path(os.path.abspath(__file__)).parent.joinpath(f'{filename}.gif'),
        [np.array(img) for i, img in enumerate(images)],
        fps=10
    )

    log.info("done")


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
    train_and_create_gif(
        jsp_instance=jsp
    )
