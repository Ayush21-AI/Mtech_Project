"""Visualization helpers for CIFAR-10 exploratory data analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _reshape_for_display(image_flat: np.ndarray) -> np.ndarray:
    """Reshape a CIFAR-10 flat (3072,) vector to (32, 32, 3) RGB."""
    return image_flat.reshape(3, 32, 32).transpose(1, 2, 0)


def show_unique_images(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
) -> tuple[list[int], Figure]:
    """Display the first occurrence of each class label.

    Parameters
    ----------
    images: np.ndarray
        Flat image matrix of shape ``(N, 3072)``.
    labels: np.ndarray
        Integer label vector of shape ``(N,)``.
    class_names: list[str] | None
        Human-readable class names. If ``None``, indices are used.
    figsize: tuple[float, float]
        Matplotlib figure size.
    save_path: str | Path | None
        Optional path to save the figure PNG.

    Returns
    -------
    tuple[list[int], Figure]
        Indices of the unique images and the matplotlib figure.
    """
    if class_names is None:
        class_names = [str(i) for i in range(int(labels.max()) + 1)]

    unique_labels: list[int] = []
    unique_indices: list[int] = []

    fig = plt.figure(figsize=figsize)
    subplot_index = 0
    for i, label in enumerate(labels):
        if label in unique_labels:
            continue

        ax = fig.add_subplot(2, 5, subplot_index + 1)
        ax.imshow(_reshape_for_display(images[i]), interpolation="nearest")
        ax.set_title(class_names[label])
        ax.axis("off")

        unique_labels.append(label)
        unique_indices.append(i)
        subplot_index += 1

        if len(unique_labels) == len(class_names):
            break

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return unique_indices, fig


def show_images(
    images: np.ndarray,
    labels: np.ndarray,
    indices: list[int] | np.ndarray | None = None,
    class_names: list[str] | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
) -> Figure:
    """Display images at the requested indices.

    Parameters
    ----------
    images: np.ndarray
        Flat image matrix of shape ``(N, 3072)``.
    labels: np.ndarray
        Integer label vector of shape ``(N,)``.
    indices: list[int] | np.ndarray | None
        Indices to display. If ``None``, displays the first 10 images.
    class_names: list[str] | None
        Human-readable class names.
    figsize: tuple[float, float]
        Matplotlib figure size.
    save_path: str | Path | None
        Optional path to save the figure PNG.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if indices is None:
        indices = list(range(min(10, len(images))))
    indices = list(indices)

    if class_names is None:
        class_names = [str(i) for i in range(int(labels.max()) + 1)]

    fig = plt.figure(figsize=figsize)
    for subplot_index, image_index in enumerate(indices):
        ax = fig.add_subplot(2, 5, subplot_index + 1)
        ax.imshow(_reshape_for_display(images[image_index]))
        ax.set_title(class_names[labels[image_index]])
        ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def drop_color_channel(
    images: np.ndarray,
    labels: np.ndarray,
    indices: list[int] | np.ndarray,
    drop_color: int,
    class_names: list[str] | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
) -> Figure:
    """Zero out one RGB channel and display the resulting images.

    Parameters
    ----------
    images: np.ndarray
        Flat image matrix of shape ``(N, 3072)``.
    labels: np.ndarray
        Integer label vector of shape ``(N,)``.
    indices: list[int] | np.ndarray
        Indices of images to visualize.
    drop_color: int
        Channel to drop: ``0`` red, ``1`` green, ``2`` blue.
    class_names: list[str] | None
        Human-readable class names.
    figsize: tuple[float, float]
        Matplotlib figure size.
    save_path: str | Path | None
        Optional path to save the figure PNG.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    channel_names = {0: "red", 1: "green", 2: "blue"}
    if drop_color not in channel_names:
        raise ValueError("drop_color must be 0 (red), 1 (green), or 2 (blue)")

    indices = list(indices)
    modified_images = images[indices].copy()
    channel_start = drop_color * 1024
    channel_end = channel_start + 1024
    modified_images[:, channel_start:channel_end] = 0

    fig = show_images(
        modified_images,
        labels[indices],
        indices=list(range(len(indices))),
        class_names=class_names,
        figsize=figsize,
    )
    fig.suptitle(f"Channel dropped: {channel_names[drop_color]}", y=1.02)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def display_color_hists(
    images: np.ndarray,
    labels: np.ndarray,
    indices: list[int] | np.ndarray,
    class_names: list[str] | None = None,
    bins: int = 32,
    figsize: tuple[float, float] = (15, 30),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot per-channel histograms for selected images.

    Parameters
    ----------
    images: np.ndarray
        Flat image matrix of shape ``(N, 3072)``.
    labels: np.ndarray
        Integer label vector of shape ``(N,)``.
    indices: list[int] | np.ndarray
        Indices of images to visualize (one row per image).
    class_names: list[str] | None
        Human-readable class names.
    bins: int
        Number of histogram bins.
    figsize: tuple[float, float]
        Matplotlib figure size.
    save_path: str | Path | None
        Optional path to save the figure PNG.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    indices = list(indices)
    if class_names is None:
        class_names = [str(i) for i in range(int(labels.max()) + 1)]

    fig = plt.figure(figsize=figsize)
    for row, image_index in enumerate(indices):
        label = class_names[labels[image_index]]
        channels = [
            ("red", images[image_index][:1024]),
            ("green", images[image_index][1024:2048]),
            ("blue", images[image_index][2048:]),
        ]
        for col, (channel_name, values) in enumerate(channels):
            ax = fig.add_subplot(len(indices), 3, row * 3 + col + 1)
            ax.hist(values, bins=bins, color=channel_name, edgecolor="black")
            ax.set_title(f"{channel_name}: {label}")
            ax.set_xlim(0, 255)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_class_distribution(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    class_names: list[str] | None = None,
    figsize: tuple[float, float] = (12, 5),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot side-by-side class counts for train and test sets.

    Parameters
    ----------
    train_labels: np.ndarray
        Training labels.
    test_labels: np.ndarray
        Test labels.
    class_names: list[str] | None
        Human-readable class names.
    figsize: tuple[float, float]
        Matplotlib figure size.
    save_path: str | Path | None
        Optional path to save the figure PNG.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if class_names is None:
        num_classes = max(int(train_labels.max()) + 1, int(test_labels.max()) + 1)
        class_names = [str(i) for i in range(num_classes)]

    train_counts = np.bincount(train_labels, minlength=len(class_names))
    test_counts = np.bincount(test_labels, minlength=len(class_names))

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, train_counts, width, label="train")
    ax.bar(x + width / 2, test_counts, width, label="test")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("CIFAR-10 class distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
