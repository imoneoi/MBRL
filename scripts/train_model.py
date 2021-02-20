import argparse

import numpy as np

from model.nn_mlp_model import NNMlpModel
from algorithm.model_trainer import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="static/datasets/halfcheetah_random.npz")
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


if __name__ == "__main__":
    main()
