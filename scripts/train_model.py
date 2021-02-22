import argparse
import os

import numpy as np
import torch

from model.nn_mlp_model import NNMlpModel
from model.linear_kernel_model import LinearKernelModel
from algorithm.model_trainer import train_model


def main():
    # set single thread
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="static/datasets/halfcheetah_random.npz")
    parser.add_argument("--out_dir", type=str, default="static/models")
    parser.add_argument("--model-type", type=str, default="mlp")
    parser.add_argument("--num-epochs", type=int, default=100)

    args = parser.parse_args()

    # load dataset
    dataset = np.load(args.dataset)

    # create model
    if args.model_type == "mlp":
        model_fn = NNMlpModel
    elif args.model_type == "linear":
        model_fn = LinearKernelModel
    else:
        raise NotImplementedError

    model = model_fn(
        dataset["obs"].shape[-1],
        dataset["act"].shape[-1]
    )

    # train model
    train_model(model, dataset, num_epochs=args.num_epochs)

    # save model
    torch.save(model.state_dict(), os.path.join(args.out_dir, "{}_{}.checkpoint".format(
        os.path.basename(args.dataset),
        args.model_type
    )))


if __name__ == "__main__":
    main()
