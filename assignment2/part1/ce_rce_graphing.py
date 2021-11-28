import numpy as np
import os
import argparse
import json


def ce(accuracies, resnet18_acc):
    pass


def rce(accuracies, resnet18_acc):
    pass


if __name__ == "__main__":
    models = ["vgg11", "vgg11_bn", "resnet18", "resnet34", "densenet121"]

    model_results = os.listdir("results/")
    accuracies = {}

    # loop over results files and load into memory
    for m in model_results:
        path = os.path.join("results", m)
        model_name = m[: m.rfind("_")]
        print(f"loading {model_name} data")
        with open(path, "r") as f:
            accuracies[model_name] = json.load(f)
    print(accuracies)

    # calculate ce and rce

    # plot results
    # resnet18 plot

    # other model plots

