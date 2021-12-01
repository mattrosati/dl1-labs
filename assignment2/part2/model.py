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
# Date Adapted: 2021-11-11
###############################################################################

import math
from numpy.random.mtrand import random
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """

    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.w = torch.nn.Parameter(
            torch.zeros(self.hidden_dim + self.embed_dim, 4 * self.hidden_dim)
        )
        self.b = torch.nn.Parameter(torch.zeros(4 * self.hidden_dim))

        self.mode = "no_cache"

        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        factor = 1 / math.sqrt(self.hidden_dim)
        for _, params in self.named_parameters():
            nn.init.uniform_(params, a=-1 * factor, b=factor)
        with torch.no_grad():
            self.b[2 * self.hidden_dim : 3 * self.hidden_dim] += 1
        print("initialized all parameters")

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, embedding size].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        #
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        time, batch_size, _ = embeds.shape
        out = torch.zeros(time, batch_size, self.hidden_dim).to(self.w.device)

        if self.mode == "no_cache":
            self.h = torch.zeros(batch_size, self.hidden_dim).to(self.w.device)
            self.c = torch.zeros(batch_size, self.hidden_dim).to(self.w.device)

        for time in range(time):

            # compute products
            x = embeds[time, ...]
            data = torch.cat((self.h, x), dim=-1)
            big_mult = torch.matmul(data, self.w) + self.b
            # big_mult = torch.matmul(self.h, self.wh) + torch.matmul(x, self.wx) + self.b
            g, i, f, o = torch.chunk(big_mult, 4, dim=1)

            # slice to get various gate values
            g = torch.tanh(g)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            self.c = g * i + self.c * f
            self.h = torch.tanh(self.c) * o
            out[time, ...] = self.h

        print(out.shape)
        return out

        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """

    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.vocabulary_size = args.vocabulary_size
        self.embedding_size = args.embedding_size
        self.lstm_hidden_dim = args.lstm_hidden_dim

        self.e = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.lstm = LSTM(self.lstm_hidden_dim, self.embedding_size)
        self.linear = nn.Linear(self.lstm_hidden_dim, self.vocabulary_size)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.e(x)
        x = self.lstm(x)
        x = self.linear(x)
        return x
        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=4, sample_length=30, temperature=0.0):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # changeeeeeeeeee
        samples = torch.randint(
            low=0, high=self.vocabulary_size, size=(sample_length, batch_size)
        )

        for i in range(1, sample_length):
            # run network on sampled characters
            out = self.forward(samples[i - 1, :].unsqueeze(dim=-1))
            print(out.shape)

            # perform softmax or argmax sampling
            if temperature > 0:
                softie = torch.softmax(out / temperature, dim=-1).squeeze()
                pred = torch.multinomial(softie, batch_size, replacement=True)
            else:
                # no need for softmax because it's a map to monotonically increasing func
                pred = out.argmax(dim=-1)

            # add results
            samples[i, :] = pred.squeeze()
            if i == 1:
                self.mode = "generate"

        samples = samples.t().tolist()
        return samples

        #######################
        # END OF YOUR CODE    #
        #######################


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument(
        "--vocabulary_size", type=int, default=30, help="Length of an input sequence"
    )
    parser.add_argument(
        "--lstm_hidden_dim",
        type=int,
        default=5,
        help="Number of hidden units in the LSTM",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=10,
        help="Dimensionality of the embeddings.",
    )

    args = parser.parse_args()
    model = LSTM(5, 4)
    out = model.forward(nn.init.uniform_(torch.zeros(2, 3, 4)))
    # print(out)
    # print(out.shape)
    text_gen = TextGenerationModel(args)
    print(len(text_gen.sample()[0]))

