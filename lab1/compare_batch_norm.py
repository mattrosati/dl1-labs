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
# Date Created: 2021-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch as mlp
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Run all hyperparameter configurations as requested
    results = []
    lr = 0.1
    batch_size = 128
    epochs = 20
    seed = 42
    data_dir = "data/"
    hidden_dims_array = [[128], [256, 128], [512, 256, 128]]

    print("Training all models")
    for i, hidden_dims in enumerate(hidden_dims_array):
        print("Without BatchNorm")
        model, val_accuracies, test_accuracy, logging_dict = mlp.train(
            hidden_dims, lr, False, batch_size, epochs, seed, data_dir
        )
        results_nobatch = {
            "info": model.info,
            "val_acc": val_accuracies,
            "test_acc": test_accuracy,
            "other": logging_dict,
        }

        print("With BatchNorm")
        model, val_accuracies, test_accuracy, logging_dict = mlp.train(
            hidden_dims, lr, True, batch_size, epochs, seed, data_dir
        )
        results_batch = {
            "info": model.info,
            "val_acc": val_accuracies,
            "test_acc": test_accuracy,
            "other": logging_dict,
        }
        results += [results_nobatch, results_batch]

    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file

    with open(results_filename, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    with open(results_filename, "rb") as handle:
        data = pickle.load(handle)

    f, axs = plt.subplots(2, 3, sharey="row")
    f.suptitle("Hyperparameter Configurations Results")
    plt.subplots_adjust(hspace=0.5, left=0.05, right=0.95)
    for j in range(0, len(data), 2):
        i = j // 2
        epochs = data[0]["other"]["epochs"]
        axs[0, i].plot(
            epochs, data[j]["other"]["train_acc"], "b-", label="No batch norm",
        )
        axs[0, i].plot(
            epochs, data[j + 1]["other"]["train_acc"], "b--", label="With batch norm",
        )

        axs[1, i].plot(
            epochs, data[j]["val_acc"], "r-", label="No batch norm",
        )
        axs[1, i].plot(
            epochs, data[j + 1]["val_acc"], "r--", label="With batch norm",
        )

        n_hidden = data[2 * i]["info"]["hidden_layers"]

        axs[0, i].set_title(
            "Training Accuracies per Epoch\n" + f"for {n_hidden} hidden dimensions"
        )
        axs[1, i].set_title(
            "Validation Accuracies per Epoch\n" + f"for {n_hidden} hidden dimensions"
        )
        axs[0, i].set_xlabel("Epochs")
        axs[0, i].set_ylabel("Accuracy")
        axs[1, i].set_xlabel("Epochs")
        axs[1, i].set_ylabel("Accuracy")

    axs[0, 2].legend(loc="lower right")
    axs[1, 2].legend(loc="lower right")
    # f.tight_layout(pad=0.2)
    plt.show()

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    # Feel free to change the code below as you need it.
    FILENAME = "results_batchnorm.txt"
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)
