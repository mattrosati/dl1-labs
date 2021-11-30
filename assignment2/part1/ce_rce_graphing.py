import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
    values = {}
    for c in accuracies.keys():
        if c != "clean":
            values[c] = np.sum(1 - np.array(accuracies[c])) / np.sum(
                1 - np.array(resnet18_acc[c])
            )
    return values


def rce(accuracies, resnet18_acc):
    values = {}
    for c in accuracies.keys():
        if c != "clean":
            model_err = 1 - np.array(accuracies[c])
            clean_err = 1 - np.array(accuracies["clean"])
            norm_err = 1 - np.array(resnet18_acc[c])
            norm_clean_err = 1 - np.array(resnet18_acc["clean"])
            values[c] = np.sum(model_err - clean_err) / np.sum(
                norm_err - norm_clean_err
            )
    return values


if __name__ == "__main__":
    models = ["vgg11", "vgg11_bn", "resnet18", "resnet34", "densenet121"]
    corruptions = {
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
            f, ax = plt.subplots()
            for c in accuracies[key].keys():
                if c == "clean":
                    ax.plot(
                        np.arange(1, 6),
                        np.repeat(accuracies[key][c], 5),
                        "k--",
                        label="Baseline",
                    )
                else:
                    ax.plot(
                        np.arange(1, 6),
                        accuracies[key][c],
                        color=colors.pop(0),
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
            f.tight_layout()
            plt.show()

    # plot rce and ce results
    f, axs = plt.subplots(1, 2, sharey=True)
    color_map = cm.get_cmap("Purples")
    colors = color_map(np.linspace(0, 1, num=4))
    bar_width = 0.2
    y_ticks = np.arange(len(corruptions))
    for i, key in enumerate(metrics.keys()):
        j = 0
        for model in metrics[key]:
            axs[i].barh(
                y=y_ticks + j * bar_width,
                width=list(metrics[key][model].values()),
                label=model,
                height=bar_width,
                edgecolor="black",
                color=colors[j],
            )
            j += 1
            axs[i].set_yticklabels([corruptions[k] for k in metrics[key][model].keys()])
        axs[i].set_yticks(y_ticks + bar_width * (len(corruptions) - 1) / 2)
        if key == "ce":
            axs[i].set_title("CE values (normalized with ResNet18 accuracies)")
        else:
            axs[i].set_title("RCE values (normalized with ResNet18 accuracies)")
    axs[-1].legend()
    f.suptitle("CE and RCE metrics of various models and data corruption functions")
    plt.show()

