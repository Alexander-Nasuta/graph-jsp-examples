import numpy as np
import stable_baselines3 as sb3

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from graph_jsp_env.disjunctive_graph_logger import log


def jss_env_basic_ppo_example(jsp_instance: np.ndarray, total_timesteps: int = 20_000, n_eval_episodes: int = 10) \
        -> None:
    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp_instance,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=True,
        action_mode='task',  # alternative 'job'
        dtype='float32'
    )

    env = sb3.common.monitor.Monitor(env)
    model = sb3.PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    log.info("training the model")
    model.learn(total_timesteps=total_timesteps)

    log.info("evaluating the model")
    mean, std = sb3.common.evaluation.evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=False)

    log.info(f"Model mean reward: {mean:.2f}, std: {std:.2f}")


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
    jss_env_basic_ppo_example(jsp_instance=jsp)
