################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-17
################################################################################
from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import *
from networks import *


def permute_indices(molecules: Batch) -> Batch:
    """permute the atoms within a molecule, but not across molecules

    Args:
        molecules: batch of molecules from pytorch geometric

    Returns:
        permuted molecules
    """
    # Permute the node indices within a molecule, but not across them.
    ranges = [
        (i, j) for i, j in zip(molecules.ptr.tolist(), molecules.ptr[1:].tolist())
    ]
    permu = torch.cat([torch.arange(i, j)[torch.randperm(j - i)] for i, j in ranges])

    n_nodes = molecules.x.size(0)
    inits = torch.arange(n_nodes)
    # For the edge_index to work, this must be an inverse permutation map.
    translation = {k: v for k, v in zip(permu.tolist(), inits.tolist())}

    permuted = deepcopy(molecules)
    permuted.x = permuted.x[permu]
    # Below is the identity transform, by construction of our permutation.
    permuted.batch = permuted.batch[permu]
    permuted.edge_index = (
        permuted.edge_index.cpu()
        .apply_(translation.get)
        .to(molecules.edge_index.device)
    )
    return permuted


def compute_loss(
    model: nn.Module, molecules: Batch, criterion: Callable
) -> torch.Tensor:
    """use the model to predict the target determined by molecules. loss computed by criterion.

    Args:
        model: trainable network
        molecules: batch of molecules from pytorch geometric
        criterion: callable which takes a prediction and the ground truth 

    Returns:
        loss

    TODO: 
    - conditionally compute loss based on model type
    - make sure there are no warnings / errors
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    labels = get_labels(molecules)
    if model.info["name"] == "MLP":
        features = get_mlp_features(molecules)
        out = model.forward(features)
    elif model.info["name"] == "GNN":
        features = get_node_features(molecules)
        out = model.forward(
            features,
            molecules.edge_index,
            molecules.edge_attr.argmax(dim=-1),
            molecules.batch,
        )
    else:
        print("something is wonky with network name")

    out = out.squeeze()
    loss = criterion(out, labels)
    #######################
    # END OF YOUR CODE    #
    #######################
    return loss


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, criterion: Callable, permute: bool
) -> float:
    """
    Performs the evaluation of the model on a given dataset.

    Args:
        model: trainable network
        data_loader: The data loader of the dataset to evaluate.
        criterion: loss module, i.e. torch.nn.MSELoss()
        permute: whether to permute the atoms within a molecule
    Returns:
        avg_loss: scalar float, the average loss of the model on the dataset.

    Hint: make sure to return the average loss of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    
    TODO: conditionally permute indices
          calculate loss
          average loss independent of batch sizes
          make sure the model is in the correct mode
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    avg_loss = 0
    model.eval()
    n_batches = len(data_loader)
    for batch in data_loader:
        if permute:
            batch = permute_indices(batch)
        loss_batch = compute_loss(model, batch, criterion)
        avg_loss += loss_batch.item()
    avg_loss /= n_batches
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_loss


def train(
    model: nn.Module, lr: float, batch_size: int, epochs: int, seed: int, data_dir: str
):
    """a full training cycle of an mlp / gnn on qm9.

    Args:
        model: a differentiable pytorch module which estimates the U0 quantity
        lr: learning rate of optimizer
        batch_size: batch size of molecules
        epochs: number of epochs to optimize over
        seed: random seed
        data_dir: where to place the qm9 data

    Returns:
        model: the trained model which performed best on the validation set
        test_loss: the loss over the test set
        permuted_test_loss: the loss over the test set where atomic indices have been permuted
        val_losses: the losses over the validation set at every epoch
        logging_info: general object with information for making plots or whatever you'd like to do with it

    TODO:
    - Implement the training of both the mlp and the gnn in the same function
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')
    
    Hint: you can save your best model by deepcopy-ing it.
    """
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Loading the dataset
    train, valid, test = get_qm9(data_dir, model.device)
    train_dataloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        exclude_keys=["pos", "idx", "z", "name"],
    )
    valid_dataloader = DataLoader(
        valid, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )
    test_dataloader = DataLoader(
        test, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    device = model.device
    # TODO: Initialize loss module and optimizer
    loss_module = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    # TODO: Training loop including validation, using evaluate_model
    # TODO: Do optimization, we used adam with amsgrad. (many should work)
    model_temp = model
    n_batches_train = len(train_dataloader)
    n_batches_val = len(valid_dataloader)

    losses = {"train": np.zeros(epochs), "val": np.zeros(epochs)}
    modes = ["train", "val"]

    best_loss = np.inf

    for epoch in range(epochs):
        for m in modes:
            if m == "train":
                model_temp.train()
                print("Training epoch:", epoch + 1)
                loader = train_dataloader
                n_batch = n_batches_train
            elif m == "val":
                model_temp.eval()
                print("Validation of epoch", epoch + 1)
                loader = valid_dataloader
                n_batch = n_batches_val
            for batch in tqdm(loader, unit="batch"):
                # move to cuda if available
                batch = batch.to(device)

                # run forward and compute loss
                loss_batch = compute_loss(model_temp, batch, loss_module)
                losses[m][epoch] += loss_batch.item()

                # run backwards and update params
                if m == "train":
                    optimizer.zero_grad()
                    loss_batch.backward()
                    optimizer.step()

            # average loss over batches
            losses[m][epoch] = losses[m][epoch] / n_batch

            if m == "val":
                if losses[m][epoch] < best_loss:
                    print(
                        "New best model found, new best loss is", losses[m][epoch],
                    )
                    best_model = deepcopy(model_temp)
                    best_loss = losses[m][epoch]
    # TODO: Test best model
    print("Testing model...")
    test_loss = evaluate_model(best_model, test_dataloader, loss_module, permute=False)
    print("Best model test loss:", test_loss)
    # TODO: Test best model against permuted indices
    permuted_test_loss = evaluate_model(
        best_model, test_dataloader, loss_module, permute=True
    )
    print("Permuted best model test loss:", permuted_test_loss)
    # TODO: Add any information you might want to save for plotting
    val_losses = losses["val"]
    logging_info = losses["train"]

    #######################
    # END OF YOUR CODE    #
    #######################
    return model, test_loss, permuted_test_loss, val_losses, logging_info


def main(**kwargs):
    """main handles the arguments, instantiates the correct model, tracks the results, and saves them."""
    which_model = kwargs.pop("model")
    mlp_hidden_dims = kwargs.pop("mlp_hidden_dims")
    gnn_hidden_dims = kwargs.pop("gnn_hidden_dims")
    gnn_num_blocks = kwargs.pop("gnn_num_blocks")

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if which_model == "mlp":
        model = MLP(FLAT_INPUT_DIM, mlp_hidden_dims, 1)
    elif which_model == "gnn":
        model = GNN(
            n_node_features=Z_ONE_HOT_DIM,
            n_edge_features=EDGE_ATTR_DIM,
            n_hidden=gnn_hidden_dims,
            n_output=1,
            num_convolution_blocks=gnn_num_blocks,
        )
    else:
        raise NotImplementedError("only mlp and gnn are possible models.")

    model.to(device)
    model, test_loss, permuted_test_loss, val_losses, logging_info = train(
        model, **kwargs
    )

    # plot the loss curve, etc. below.

    f, ax = plt.subplots()
    ax.plot(np.arange(1, len(val_losses) + 1), val_losses, "-bo", label="Validation")
    ax.plot(np.arange(1, len(logging_info) + 1), logging_info, "-go", label="Training")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Loss")
    ax.set_title("Average Loss per Epoch for " + which_model.upper())
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model",
        default="mlp",
        type=str,
        choices=["mlp", "gnn"],
        help="Select between training an mlp or a gnn.",
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        default=[128, 128, 128, 128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the mlp. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--gnn_hidden_dims",
        default=64,
        type=int,
        help="Hidden dimensionalities to use inside the mlp. The same number of hidden features are used at every layer.",
    )
    parser.add_argument(
        "--gnn_num_blocks",
        default=2,
        type=int,
        help="Number of blocks of GNN convolutions. A block may include multiple different kinds of convolutions (see GNN comments)!",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")

    # Technical
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the qm9 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
