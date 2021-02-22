from multiprocessing import connection, context, Pipe, cpu_count
from select import select
import random

import numpy as np
import gym
from tqdm import tqdm
import torch

from policy.base_policy import BasePolicy
from utils.cloudpickle_wrapper import CloudpickleWrapper


def worker_(
        pipe: connection.Connection,
        env_wrapper: CloudpickleWrapper,
        policy_wrapper: CloudpickleWrapper,
        num_data_points: int,
        seed: int
):
    # Unwrap
    env = env_wrapper.data
    policy = policy_wrapper.data

    # seed all RNGs
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # collect
    obs = env.reset()
    for step in range(num_data_points):
        act = policy.infer(obs)
        next_obs, rew, done, info = env.step(act)

        # send data
        pipe.send({
            "obs": obs,
            "act": act,
            "rew": rew,
            "done": done,
            "next_obs": next_obs
        })

        # update obs
        obs = next_obs
        # reset if done
        if done:
            obs = env.reset()


def collect_data(
        env: gym.Env,
        policy: BasePolicy,
        num_data_points: int,

        n_jobs: int = cpu_count()
):
    # Launch subprocesses
    processes = []
    pipes = []

    for proc_idx in range(n_jobs):
        parent_remote, child_remote = Pipe()
        args = (
            child_remote,
            CloudpickleWrapper(env),
            CloudpickleWrapper(policy),
            num_data_points // n_jobs,
            proc_idx  # independent seed for every worker
        )
        process = context.Process(target=worker_, args=args, daemon=True)

        pipes.append(parent_remote)
        processes.append(process)

    # Start subprocesses
    for proc in processes:
        proc.start()

    # Receive dataset
    t = tqdm(total=num_data_points)
    dataset = {}
    while True:
        # Check process alive
        num_alive = sum([proc.is_alive() for proc in processes])
        if not num_alive:
            break

        # Wait for read
        select([pipe.fileno() for pipe in pipes], [], [], 0.1)

        # Read pipes
        for pipe in pipes:
            if not pipe.poll():
                continue

            # Receive data point
            try:
                data_point = pipe.recv()
                for k, v in data_point.items():
                    dataset.setdefault(k, [])
                    dataset[k].append(v)

                t.update(1)
            except EOFError:
                # Ignore EOF (when process died)
                pass

    # Convert dataset to numpy
    dataset = {k: np.array(v) for k, v in dataset.items()}

    return dataset
