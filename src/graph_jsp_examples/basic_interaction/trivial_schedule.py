import time

import numpy as np

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from graph_jsp_env.disjunctive_graph_logger import log


def trivial_schedule(jsp_instance: np.ndarray) -> None:
    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp_instance,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=True,
        action_mode='task',  # alternative 'job'
        dtype='float32'
    )

    done = False
    score = 0

    iteration_count = 0
    start = time.perf_counter()
    for i in range(env.total_tasks_without_dummies):
        n_state, reward, done, info = env.step(i)
        score += reward
        iteration_count += 1

    end = time.perf_counter()

    env.render(show=["gantt_console", "graph_console"])

    log.info(f"score: {score}")
    total_duration = end - start
    log.info(f"total duration: {total_duration:2f} sec")
    log.info(f"average iteration duration: {total_duration / iteration_count:4f} sec")


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
    trivial_schedule(jsp_instance=jsp)
