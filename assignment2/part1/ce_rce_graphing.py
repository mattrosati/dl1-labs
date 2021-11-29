import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt


def ce(accuracies, resnet18_acc):
    pass


def rce(accuracies, resnet18_acc):
    pass


if __name__ == "__main__":
    models = ["vgg11", "vgg11_bn", "resnet18", "resnet34", "densenet121"]
    corruptions = {
        "clean": "Baseline",
        "gnoise": "Gaussian Noise",
        "gblur": "Gaussian Blur",
        "contrast": "Contrast Reduction",
        "jpg": "JPEG Compression",
    }
    colors = ["green", "cyan", "magenta", "purple"]

    model_results = os.listdir("results/")
    accuracies = {}

    # loop over results files and load into memory
    for m in model_results:
        path = os.path.join("results", m)
        model_name = m[: m.rfind("_")]
        print(f"loading {model_name} data")
        with open(path, "r") as f:
            accuracies[model_name] = json.load(f)

    # calculate ce and rce, plot resnet18 results
    metrics = {}
    metrics["ce"] = {}
    metrics["rce"] = {}
    for val in accuracies.keys():
        if val != "resnet18":
            ce = tbd
        else:
            f, ax = plt.subplots(1, 1)
            for c in accuracies[val].keys():
                if c == "clean":
                    ax.plot(
                        accuracies[val][c], np.arange(1, 6), "k--", label=corruptions[c]
                    )
                else:
                    ax.plot(
                        accuracies[val][c],
                        np.arange(1, 6),
                        color=next(iter(colors)),
                        linestyle="solid",
                        label=corruptions[c],
                    )
            plt.show()

    # plot rce and ce results

