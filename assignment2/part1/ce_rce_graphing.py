import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt


def load_results(file_list):
    accuracies = {}

    # loop over file list and load into memory
    for m in file_list:
        path = os.path.join("results", m)
        model_name = m[: m.rfind("_")]
        print(f"loading {model_name} data")
        with open(path, "r") as f:
            accuracies[model_name] = json.load(f)

    return accuracies


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
    colors = ["green", "blue", "red", "purple"]

    model_results = os.listdir("results/")
    accuracies = load_results(model_results)

    # calculate ce and rce, plot resnet18 results
    metrics = {}
    metrics["ce"] = {}
    metrics["rce"] = {}
    for key in accuracies.keys():
        if key != "resnet18":
            metrics["ce"][key] = ce(accuracies[key], accuracies["resnet18"])
            metrics["rce"][key] = rce(accuracies[key], accuracies["resnet18"])
        else:
            f, ax = plt.subplots(1, 1)
            for c in accuracies[key].keys():
                if c == "clean":
                    ax.plot(
                        np.arange(1, 6),
                        np.repeat(accuracies[key][c], 5),
                        "k--",
                        label=corruptions[c],
                    )
                else:
                    ax.plot(
                        np.arange(1, 6),
                        accuracies[key][c],
                        color=colors.pop(),
                        marker="o",
                        linestyle="solid",
                        label=corruptions[c],
                    )
            ax.set_xlabel("Corruption Severity")
            ax.set_ylabel("Top-1 Accuracy")
            ax.set_title(
                "Test Accuracy of ResNet18 with various corruption functions (CIFAR10 Dataset)"
            )
            ax.set_xticks(np.arange(1, 6))
            ax.legend()
            plt.show()

    # plot rce and ce results

