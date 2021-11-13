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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
        This class implements a Multi-layer Perceptron in NumPy.
        It handles the different layers and parameters of the model.
        Once initialized an MLP object can perform forward and backward.
        """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
                Initializes MLP object.

                Args:
                    n_inputs: number of inputs.
                    n_hidden: list of ints, specifies the number of units
                                    in each linear layer. If the list is empty, the MLP
                                        will not have any linear layers, and the model
                                        will simply perform a multinomial logistic regression.
                    n_classes: number of classes of the classification problem.
                                         This number is required in order to specify the
                                         output dimensions of the MLP

                TODO:
                Implement initialization of the network.
                """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        dims = [n_inputs] + n_hidden + [n_classes]
        layers = len(dims) - 1

        self.modules = []

        self.modules.append(
            LinearModule(in_features=dims[0], out_features=dims[1], input_layer=True)
        )
        for i in np.arange(1, layers):
            self.modules.append(ReLUModule())
            self.modules.append(
                LinearModule(
                    in_features=dims[i], out_features=dims[i + 1], input_layer=False
                )
            )
        self.modules.append(SoftMaxModule())

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
                Performs forward pass of the input. Here an input tensor x is transformed through
                several layer transformations.

                Args:
                    x: input to the network
                Returns:
                    out: outputs of the network

                TODO:
                Implement forward pass of the network.
                """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        x = x.reshape((x.shape[0], -1))
        print(x.shape)
        for i in self.modules:
            x = i.forward(x)
        out = x

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
                Performs backward pass given the gradients of the loss.

                Args:
                    dout: gradients of the loss

                TODO:
                Implement backward pass of the network.
                """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for i in reversed(self.modules):
            dout = i.backward(dout)

        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
                Remove any saved tensors for the backward pass from any module.
                Used to clean-up model from any remaining input data when we want to save it.

                TODO:
                Iterate over modules and call the 'clear_cache' function.
                """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for i in self.modules:
            i.clear_cache()

        #######################
        # END OF YOUR CODE    #
        #######################


if __name__ == "__main__":
    # Command line arguments
    import argparse

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

    x = MLP(3 * 1024, [100, 64], 10)
    y = np.random.rand(128, 3, 32, 32)
    print(x.forward(y).shape)
    loss = np.random.rand(128, 10)
    x.backward(loss)
    print(x.modules[0].grads["bias"].shape)
    print(x.modules[0].params["bias"].shape)
    print(x.modules[2].grads["bias"].shape)
    print(x.modules[2].params["bias"].shape)

