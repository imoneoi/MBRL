import argparse
import os

import numpy as np
import torch

from model.nn_mlp_model import NNMlpModel
from algorithm.model_trainer import train_model


def main():
    # set single thread
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="static/datasets/halfcheetah_random.npz")
    parser.add_argument("--out_dir", type=str, default="static/models")
    parser.add_argument("--num-epochs", type=int, default=100)

    args = parser.parse_args()

    # load dataset
    dataset = np.load(args.dataset)

    # create model
    model = NNMlpModel(
        dataset["obs"].shape[-1],
        dataset["act"].shape[-1]
    )

    # train model
    train_model(model, dataset, num_epochs=args.num_epochs)

    # save model
    torch.save(model.state_dict(), os.path.join(args.out_dir, os.path.basename(args.dataset) + ".checkpoint"))


if __name__ == "__main__":
    main()
