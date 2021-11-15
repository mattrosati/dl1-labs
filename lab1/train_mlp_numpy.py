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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch
import matplotlib.pyplot as plt


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    predicted_labels = np.argmax(predictions, axis=-1)
    accuracy = np.mean(predicted_labels == targets)

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    accuracy_sum = 0
    for data, targets in data_loader:
        scores = model.forward(data)
        batch_accuracy = accuracy(scores, targets)
        accuracy_sum += batch_accuracy

    avg_accuracy = accuracy_sum / len(data_loader)

    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=True
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    dims = np.prod(list(cifar10["train"][0][0].shape))
    class_num = 10

    # TODO: Initialize model and loss module
    model_temp = MLP(dims, hidden_dims, class_num)
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation
    n_batches_train = len(cifar10_loader["train"])
    n_batches_val = len(cifar10_loader["validation"])

    train_loss = np.zeros(epochs)
    train_accuracies = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    val_accuracies = np.zeros(epochs)

    best_accuracy = 0
    for epoch in range(epochs):
        print("Training epoch:", epoch + 1)
        for data, targets in tqdm(cifar10_loader["train"], unit="batch"):
            # run forward
            out = model_temp.forward(data)

            # calculate loss and accuracy
            loss_batch = loss_module.forward(out, targets)
            train_loss[epoch] += loss_batch
            batch_accuracy = accuracy(out, targets)
            train_accuracies[epoch] += batch_accuracy

            # run backward and update weights
            loss_back = loss_module.backward(out, targets)
            model_temp.backward(loss_back)
            for module in model_temp.modules[::2]:
                module.params["weight"] -= lr * module.grads["weight"]
                module.params["bias"] -= lr * module.grads["bias"]

        # average accuracy and loss over batches
        train_accuracies[epoch] = train_accuracies[epoch] / n_batches_train
        train_loss[epoch] = train_loss[epoch] / n_batches_train

        print("Validation of epoch", epoch + 1)
        for data, targets in tqdm(cifar10_loader["validation"], unit="batch"):
            out = model_temp.forward(data)
            loss_batchv = loss_module.forward(out, targets)
            val_loss[epoch] += loss_batchv
            val_accuracies[epoch] += accuracy(out, targets)

        val_accuracies[epoch] = val_accuracies[epoch] / n_batches_val
        val_loss[epoch] = val_loss[epoch] / n_batches_val

        if val_accuracies[epoch] > best_accuracy:
            print(
                "New best model found, copying it, new best accuracy is",
                val_accuracies[epoch],
            )
            model_temp.clear_cache()
            best_model = deepcopy(model_temp)
            best_accuracy = val_accuracies[epoch]

    # TODO: Test best model
    print("Testing model...")
    test_accuracy = evaluate_model(best_model, cifar10_loader["test"])
    print("Best model accuracy on test set:", test_accuracy)

    # TODO: Add any information you might want to save for plotting
    logging_dict = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_accuracies,
        "epochs": np.arange(1, epochs + 1),
    }
    model = best_model
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"',
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR10 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(logging_dict["epochs"], logging_dict["train_loss"], "b-", label="Training")
    ax1.plot(logging_dict["epochs"], logging_dict["val_loss"], "r-", label="Validation")
    ax1.set_title("Loss Values per Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epochs")
    ax1.legend()

    ax2.plot(logging_dict["epochs"], logging_dict["train_acc"], "b-", label="Training")
    ax2.plot(logging_dict["epochs"], val_accuracies, "r-", label="Validation")
    ax2.set_title("Accuracies per Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.legend()

    f.suptitle("Numpy MLP Implementation Results")
    plt.show()

