import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import List, Any

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import models


def plot_weights(model, layer_num, single_channel=False, collated=False):
    # extracting the model features at the particular layer number
    layer = list(model.parameters())[layer_num]
    # layer = model.features[layer_num]

    # checking whether the layer is convolution layer or not
    if layer.ndim > 3:
        # getting the weight tensor data
        weight_tensor = list(model.parameters())[layer_num].data

        if single_channel:
            if collated:
                plot_filters_single_channel_big(weight_tensor)
            else:
                plot_filters_single_channel(weight_tensor)

        else:
            plot_filters_multi_channel(weight_tensor)

    else:
        print("Can only visualize layers which are convolutional")


def plot_filters_single_channel_big(t):
    # setting the rows and columns
    nrows = t.shape[0] * t.shape[2]
    ncols = t.shape[1] * t.shape[3]

    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)

    npimg = npimg.T

    fig, ax = plt.subplots(figsize=(ncols / 10, nrows / 200))
    sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)
    plt.savefig('myimage.png', dpi=100)


def plot_filters_single_channel(t):
    # kernels depth * number of kernels
    nplots = t.shape[0] * t.shape[1]
    ncols = 12

    nrows = 1 + nplots // ncols

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    # looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.savefig('myimage.png', dpi=100)


def plot_filters_multi_channel(t):
    # get the number of kernals
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.tight_layout()
    plt.savefig('myimage.png', dpi=100)


def _get_class(model_tuple_list: List[Any], model_name: str, weights: Path, num_classes: int = None):
    for n, c in model_tuple_list:
        if n == model_name:
            if num_classes is not None:
                m = c(num_classes=num_classes)
            else:
                m = c()
            m.load_state_dict(torch.load(weights), strict=True)
            return m
    return None


if __name__ == '__main__':
    import_raw = importlib.import_module('src.models.backbones')
    model_classes = inspect.getmembers(import_raw, inspect.isclass)

    parser = argparse.ArgumentParser(description='Visualization of e specific layer in a neural network')
    parser.add_argument('--network_name', '-n',
                        help='Name of the network (needs to be in the src network folder)',
                        type=str,
                        choices=[n for n, _ in model_classes],
                        required=True)
    parser.add_argument('--weights', '-w',
                        help='Path to the network weights',
                        type=Path,
                        required=True)
    parser.add_argument('--output_path', '-o',
                        help='Where to save the filter images',
                        type=Path,
                        required=True)
    parser.add_argument('--layer_num', '-l',
                        help='Number of the layer you want to visualize (to find numbers print(model))',
                        type=int,
                        required=True)
    parser.add_argument('--single_channel', '-s',
                        help='Represent the image in a single channel fashion',
                        action='store_true')
    parser.add_argument('--collated', '-c',
                        help='Combine all filter to one image',
                        action='store_true')
    parser.add_argument('--num_classes', '-nc',
                        help='If you use a UNet you need to report the number of classes',
                        type=int)

    args = parser.parse_args()
    if 'UNet' in args.network_name and args.num_classes is None:
        print(f"You are using a UNet model ({args.network_name}) without specifying the num_classes")
        sys.exit(1)

    model = _get_class(model_tuple_list=model_classes, model_name=args.network_name, num_classes=args.num_classes,
                       weights=args.weights)
    plot_weights(model=model,
                 layer_num=args.layer_num, single_channel=args.single_channel, collated=args.collated)
