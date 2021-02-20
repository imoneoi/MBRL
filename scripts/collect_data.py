try:
    import local_debug_logger
except ModuleNotFoundError:
    pass

import argparse

import gym
import numpy as np

from policy.random_policy import RandomPolicy
from env.mujoco.halfcheetah_v3_custom import HalfCheetahEnv
from algorithm.data_collector import collect_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=int(1e6))
    parser.add_argument("--out", type=str, default="static/datasets/halfcheetah_random.npz")

    args = parser.parse_args()

    # env and policy
    env = HalfCheetahEnv()
    policy = RandomPolicy(env)

    # collect
    dataset = collect_data(
        env,
        policy,
        args.n
    )
    # save dataset
    np.savez(args.out, **dataset)


if __name__ == "__main__":
    main()
