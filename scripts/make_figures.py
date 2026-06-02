"""Generate the README results figure from the model metadata.

Run:  python scripts/make_figures.py   (writes docs/images/results.png)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from frauddet.config import REPO_ROOT  # noqa: E402

META = json.loads((REPO_ROOT / "models" / "metadata.json").read_text())
OUT = REPO_ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    metrics = META["metrics"]
    models = [k for k in metrics if k != "baseline"]
    pr_auc = [metrics[k]["pr_auc"] for k in models]
    colours = ["#1f77b4" if k == "xgb_classweight" else "#b0bec5" for k in models]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax[0].bar(models, pr_auc, color=colours)
    ax[0].set_title("PR-AUC by model (held-out · higher = better)")
    ax[0].set_ylim(0, 1.05)
    for i, v in enumerate(pr_auc):
        ax[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax[0].tick_params(axis="x", rotation=20)

    ens = metrics["ensemble"]
    cm = np.array([[ens["tn"], ens["fp"]], [ens["fn"], ens["tp"]]])
    im = ax[1].imshow(cm, cmap="Reds")
    ax[1].set_xticks([0, 1], ["pred legit", "pred fraud"])
    ax[1].set_yticks([0, 1], ["true legit", "true fraud"])
    for (i, j), v in np.ndenumerate(cm):
        ax[1].text(j, i, f"{v:,}", ha="center", va="center",
                   color="white" if v > cm.max() / 2 else "black", fontsize=10)
    ax[1].set_title(f"Ensemble confusion matrix (thr={ens['threshold']})")
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    tag = "SYNTHETIC data" if not META.get("data_is_real", False) else "real ULB data"
    fig.suptitle(f"Fraud Detection — held-out results ({tag})", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "results.png", dpi=130, bbox_inches="tight")
    print("saved", OUT / "results.png")


if __name__ == "__main__":
    main()
