"""Plotting helpers for Secchi MDN outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_scatterplot(
    x,
    y,
    output_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
):
    """Save a measured-vs-estimated scatterplot with a 1:1 reference line."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vmin = float(min(x.min(), y.min()))
    vmax = float(max(x.max(), y.max()))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(x, y, s=16, alpha=0.45, edgecolors="none")
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.2, color="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_three_panel_comparison(
    output_path: str | Path,
    sensor: str,
    observed,
    ours,
    maciel,
):
    """Save the observed/ours/maciel comparison panel used for validation outputs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
    panels = [
        ("Observed Secchi (m)", observed, "Our Final MDN (m)", ours, "Our vs Observed"),
        ("Observed Secchi (m)", observed, "Maciel 2023 Predicted (m)", maciel, "Maciel vs Observed"),
        ("Maciel 2023 Predicted (m)", maciel, "Our Final MDN (m)", ours, "Our vs Maciel"),
    ]
    for ax, (xlabel, x, ylabel, y, title) in zip(axes, panels):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        vmin = float(min(x.min(), y.min()))
        vmax = float(max(x.max(), y.max()))
        ax.scatter(x, y, s=16, alpha=0.45, edgecolors="none")
        ax.plot([vmin, vmax], [vmin, vmax], "--", color="black", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{sensor.upper()} {title}")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
