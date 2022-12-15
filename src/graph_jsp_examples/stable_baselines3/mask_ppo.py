import gym
import sb3_contrib

import numpy as np
import stable_baselines3 as sb3


from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from graph_jsp_env.disjunctive_graph_logger import log

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


def jss_env_mask_ppo_example(jsp_instance: np.ndarray, total_timesteps: int = 20_000) -> None:
    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp_instance,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=True,
        action_mode='task',  # alternative 'job'
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

    # NOTE: evaliation seems not to use the action mask
    # log.info("evaluating the model")
    # mean, std = sb3.common.evaluation.evaluate_policy(model, env, n_eval_episodes=10, deterministic=False)
    # log.info(f"Model mean reward: {mean:.2f}, std: {std:.2f}")


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
    jss_env_mask_ppo_example(jsp_instance=jsp)
