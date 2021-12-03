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

from datetime import datetime
import argparse
from torch.utils import data
from tqdm.auto import tqdm
import os

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel


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


def train(args):
    """
    Trains an LSTM model on a text dataset
    
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(args.seed)
    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    data_loader = DataLoader(
        dataset,
        args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=text_collate_fn,
    )
    args.vocabulary_size = dataset.vocabulary_size
    # Create model
    model = TextGenerationModel(args)
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_module = nn.CrossEntropyLoss().to(args.device)

    # make writer object
    writer = SummaryWriter("logger/")
    model_plotted = False

    # Training loop
    model = model.to(args.device)
    batch_n = len(data_loader)
    model.train()
    for i in range(args.num_epochs):

        # initialize metrics
        loss = 0
        acc = 0
        for batch, labels in tqdm(data_loader, unit="batch"):
            batch = batch.to(args.device)
            labels = labels.to(args.device)

            out = model(batch)

            out = out.view(-1, model.vocabulary_size)
            labels = labels.view(-1)
            loss_batch = loss_module(out, labels)
            loss += loss_batch.item()

            acc += accuracy(out, labels)

            # run backwards and update params
            optimizer.zero_grad()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            model.lstm.h = None
            model.lstm.c = None

        # add loss and acc to TensorBoard
        loss /= batch_n
        acc /= batch_n
        writer.add_scalar("Training Loss", loss, global_step=i + 1)
        writer.add_scalar("Training Accuracy", acc, global_step=i + 1)

        # save model at epochs 1, 5, and end to generate sentences
        if i == 0 or i == 4 or i == args.num_epochs - 1:
            os.makedirs("models/", exist_ok=True)
            book_name = "_".join(
                args.txt_file.replace("/", ".").replace("_", ".").split(".")[2:-1]
            )
            checkpoint_path = os.path.join(
                "models", "lstm_" + book_name + str(i) + ".pth"
            )
            torch.save(model.state_dict(), checkpoint_path)

    # close tensor logger
    writer.close()

    #######################
    # END OF YOUR CODE    #
    #######################


def accuracy(out, labels):
    return (out.argmax(dim=-1) == labels).float().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument(
        "--txt_file", type=str, required=True, help="Path to a .txt file to train on"
    )
    parser.add_argument(
        "--input_seq_length", type=int, default=30, help="Length of an input sequence"
    )
    parser.add_argument(
        "--lstm_hidden_dim",
        type=int,
        default=1024,
        help="Number of hidden units in the LSTM",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=256,
        help="Dimensionality of the embeddings.",
    )

    # Training
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size to train with."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--clip_grad_norm", type=float, default=5.0, help="Gradient clipping norm"
    )

    # Additional arguments. Feel free to add more arguments
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for pseudo-random number generator"
    )

    parser.add_argument(
        "--sample_length", type=int, default=30, help="Length of sampled sentence"
    )

    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Use GPU if available, else use CPU

    book_name = "_".join(
        args.txt_file.replace("/", ".").replace("_", ".").split(".")[2:-1]
    )
    for i in [0, 4, args.num_epochs - 1]:
        model_path = os.path.join("models", "lstm_" + book_name + str(i) + ".pth")
        if not os.path.isfile(model_path):
            print("Not all models present, training again")
            train(args)
        set_seed(args.seed)
        dataset = TextDataset(args.txt_file, args.input_seq_length)
        args.vocabulary_size = dataset.vocabulary_size
        model = TextGenerationModel(args)
        print(f"Found saved model at epoch {i+1}")
        print("Generating sentences:")
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        for temp in [0, 0.5, 1, 2.0]:
            print(f"Using temp {temp}")
            sampled = model.sample(
                sample_length=args.sample_length, batch_size=5, temperature=temp
            )
            for k, s in enumerate(sampled):
                print(f"Sample {k+1}")
                print(dataset.convert_to_string(s))
            model.lstm.h = None
            model.lstm.c = None
            print("//")
