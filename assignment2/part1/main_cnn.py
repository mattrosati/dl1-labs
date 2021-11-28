###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
import json
import argparse
import numpy as np
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

from augmentations import (
    gaussian_noise_transform,
    gaussian_blur_transform,
    contrast_transform,
    jpeg_transform,
)
from cifar10_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, num_classes=10):
    """
    Returns the model architecture for the provided model_name. 

    Args:
        model_name: Name of the model architecture to be returned. 
                    Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18', 
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == "debug":  # Use this model for debugging
        cnn_model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, num_classes))
    elif model_name == "vgg11":
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == "vgg11_bn":
        cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == "resnet18":
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == "densenet121":
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture "{model_name}"'
    return cnn_model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_set, val_set = get_train_validation_set(data_dir)

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True
    )

    # Initialize the optimizers and learning rate scheduler.
    # We provide a recommend setup, which you are allowed to change if interested.
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[90, 135], gamma=0.1
    )
    loss_module = nn.CrossEntropyLoss().to(device)

    # Training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.

    # init logger
    writer = SummaryWriter("logger/" + checkpoint_name)
    model_plotted = False

    # start training
    temp_model = model.to(device)

    best_accuracy = 0
    temp_accuracy = 0
    for i in tqdm(range(epochs), unit="epoch"):
        temp_model.train()
        train_epoch_loss = 0
        for batch, targets in train_loader:
            # move to cuda if available
            batch = batch.to(device)
            targets = targets.to(device)

            if not model_plotted:
                writer.add_graph(model, batch)
                model_plotted = True

            # run forward
            out = temp_model(batch)
            out = out.squeeze(dim=1)

            # do loss calc and
            loss_batch = loss_module(out, targets)
            train_epoch_loss += loss_batch.item()

            # run backwards and update params
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        scheduler.step()

        # add loss to TensorBoard
        train_epoch_loss /= len(train_loader)
        writer.add_scalar("training_loss", train_epoch_loss, global_step=i + 1)

        # average accuracy over validation set
        temp_accuracy = evaluate_model(temp_model, val_loader, device)
        writer.add_scalar("validation_accuracy", temp_accuracy, global_step=i + 1)

        # store best model
        if temp_accuracy > best_accuracy:
            model = deepcopy(temp_model)
            best_accuracy = temp_accuracy
    # close tensor logger
    writer.close()

    # save best model
    checkpoint_path = os.path.join("models", checkpoint_name + ".pth")
    torch.save(model.state_dict(), checkpoint_path)

    #######################
    # END OF YOUR CODE    #
    #######################
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    accuracy_sum = 0
    model.eval()
    for batch, targets in data_loader:
        batch = batch.to(device)
        targets = targets.to(device)
        scores = model(batch)
        batch_accuracy = (scores.argmax(dim=-1) == targets).float().mean().item()
        accuracy_sum += batch_accuracy

    accuracy = accuracy_sum / len(data_loader)
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test. 
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(seed)
    test_results = {}
    corruptions = ["clean", "gnoise", "gblur", "contrast", "jpg"]

    # loop through all corruptions and test
    for c in corruptions:
        print("Testing with corruption:", c)
        if c == "clean":
            test_set = get_test_set(data_dir)
            test_loader = data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
            test_results[c] = evaluate_model(model, test_loader, device)
        else:
            test_results[c] = []
            for i in range(1, 6):
                print("Severity:", i)
                if c == "gnoise":
                    augment = gaussian_noise_transform(severity=i)
                elif c == "gblur":
                    augment = gaussian_blur_transform(severity=i)
                elif c == "contrast":
                    augment = contrast_transform(severity=i)
                elif c == "jpg":
                    augment = jpeg_transform(severity=i)
                test_set = get_test_set(data_dir, augmentation=augment)
                test_loader = data.DataLoader(
                    test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                )
                test_results[c] += [evaluate_model(model, test_loader, device)]
            print("Done")

    #######################
    # END OF YOUR CODE    #
    #######################
    return test_results


def main(model_name, lr, batch_size, epochs, data_dir, seed):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    set_seed(seed)

    # load model class
    model = get_model(model_name, num_classes=10)

    # load or initialize model dict, move to gpu if available
    file_name = model_name + ".pth"
    os.makedirs("models/", exist_ok=True)
    model_path = os.path.join("models", file_name)
    if os.path.isfile(model_path):
        print("loading model")
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
    else:
        model = train_model(model, lr, batch_size, epochs, data_dir, model_name, device)

    # evaluate model
    results = test_model(model, batch_size, data_dir, device, seed)

    # save model results
    os.makedirs("results/", exist_ok=True)
    results_name = model_name + "_results.json"
    results_path = os.path.join("results", results_name)
    with open(results_path, "w") as fp:
        json.dump(results, fp)

    return results

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model_name", default="debug", type=str, help="Name of the model to train."
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=150, type=int, help="Max number of epochs")
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
    main(**kwargs)
