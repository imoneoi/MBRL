try:
    import local_debug_logger
except ModuleNotFoundError:
    pass

import argparse
import os

import numpy as np
import torch
import gym

from env.mujoco.halfcheetah_v3_custom import HalfCheetahEnv
from objective.mujoco.halfcheetah_v3_custom import objective

from model.nn_mlp_model import NNMlpModel
from policy.mpc_random_sample_policy import MPCRandomSamplePolicy


def main():
    # set single thread
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="static/models/halfcheetah_random.npz_mlp.checkpoint")
    parser.add_argument("--num-epochs", type=int, default=1)

    args = parser.parse_args()

    # create env
    env = gym.wrappers.TimeLimit(HalfCheetahEnv(), 1000)

    # load model
    model = NNMlpModel(
        env.observation_space.shape[0],
        env.action_space.shape[0]
    )
    model.load_state_dict(torch.load(args.model))

    # create policy
    policy = MPCRandomSamplePolicy(
        env, model, objective
    )

    # step
    total_rew_records = []
    for epoch in range(args.num_epochs):
        total_rew = 0
        obs = env.reset()
        while True:
            act = policy.infer(obs)
            next_obs, rew, done, info = env.step(act)

            total_rew += rew

            env.render()

            if done:
                break

            obs = next_obs

        total_rew_records.append(total_rew)

    print("Reward: {} +/- {}".format(np.mean(total_rew_records), np.std(total_rew_records)))


if __name__ == "__main__":
    main()
