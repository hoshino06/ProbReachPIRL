# -*- coding: utf-8 -*-
"""Combine the two drift error-validation panels for presentations."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main() -> None:
    out_dir = Path("plot/drift_error_validation")
    panels = [
        (out_dir / "panel_a_hjb_bdr.png", "(a) Physics consistency"),
        (out_dir / "panel_b_value_accuracy.png", "(b) Value-function accuracy"),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.7),
        gridspec_kw={"width_ratios": [0.88, 1.12]},
        constrained_layout=True,
    )
    for ax, (path, title) in zip(axes, panels):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
        ax.axis("off")

    path = out_dir / "drift_error_validation_combined.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
