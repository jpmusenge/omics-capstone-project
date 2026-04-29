"""
Results Analyzer Module
Reads the Weka classifier comparison results and generates
publication-quality figures:
"""

import logging
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config.settings import FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Consistent styling
CLASSIFIER_COLORS = {
    "SMO": "#2E86AB",
    "RandomForest": "#A23B72",
    "MultilayerPerceptron": "#F18F01",
    "J48": "#C73E1D",
}
CLASSIFIER_LABELS = {
    "SMO": "SMO (SVM)",
    "RandomForest": "Random Forest",
    "MultilayerPerceptron": "MLP (ANN)",
    "J48": "J48 (Decision Tree)",
}
CLASSIFIER_MARKERS = {
    "SMO": "o",
    "RandomForest": "s",
    "MultilayerPerceptron": "^",
    "J48": "D",
}


def load_results(filepath):
    """
    Load the Weka classifier comparison Excel file and extract
    the gene count from dataset names.

    Parameters
    ----------
    filepath : str
        Path to the Excel file with columns:
        Dataset, Classifier, Accuracy, AUC, Precision, Recall, F1, etc.

    Returns
    -------
    pd.DataFrame
        Results with an added 'gene_count' integer column.
    """
    df = pd.read_excel(filepath)
    logger.info(f"Loaded {len(df)} classifier runs from {filepath}")

    # Extract gene count from dataset name (e.g. 'prad_top500' -> 500)
    df["gene_count"] = df["Dataset"].apply(
        lambda x: int(re.search(r"(\d+)", x).group(1))
    )
    df = df.sort_values(["Classifier", "gene_count"])

    return df


def plot_accuracy_vs_genes(df, output_dir=FIGURES_DIR):
    """
    Line plot: accuracy vs gene count for each classifier.
    Shows the plateau/elbow where adding more genes stops helping.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame with gene_count, Classifier, Accuracy columns.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for clf in ["SMO", "RandomForest", "MultilayerPerceptron", "J48"]:
        subset = df[df["Classifier"] == clf].sort_values("gene_count")
        if subset.empty:
            continue
        ax.plot(
            subset["gene_count"],
            subset["Accuracy"] * 100,
            marker=CLASSIFIER_MARKERS[clf],
            color=CLASSIFIER_COLORS[clf],
            label=CLASSIFIER_LABELS[clf],
            linewidth=2,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    # Highlight the plateau region
    ax.axvspan(100, 500, alpha=0.08, color="#2E86AB", label="Plateau region (100\u2013500)")

    ax.set_xlabel("Number of Genes", fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy vs Gene Count", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xticks([50, 100, 200, 500, 1000, 2000])
    ax.set_ylim(92, 98.5)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, "accuracy_vs_genes.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)

    logger.info(f"Saved accuracy vs genes plot: {filepath}")
    return filepath


def plot_auc_vs_genes(df, output_dir=FIGURES_DIR):
    """
    Line plot: AUC vs gene count for each classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for clf in ["SMO", "RandomForest", "MultilayerPerceptron", "J48"]:
        subset = df[df["Classifier"] == clf].sort_values("gene_count")
        if subset.empty:
            continue
        ax.plot(
            subset["gene_count"],
            subset["AUC"],
            marker=CLASSIFIER_MARKERS[clf],
            color=CLASSIFIER_COLORS[clf],
            label=CLASSIFIER_LABELS[clf],
            linewidth=2,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    ax.axvspan(100, 500, alpha=0.08, color="#2E86AB", label="Plateau region (100\u2013500)")

    ax.set_xlabel("Number of Genes", fontsize=12)
    ax.set_ylabel("AUC (Area Under ROC Curve)", fontsize=12)
    ax.set_title("AUC vs Gene Count", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xticks([50, 100, 200, 500, 1000, 2000])
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, "auc_vs_genes.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)

    logger.info(f"Saved AUC vs genes plot: {filepath}")
    return filepath


def plot_classifier_comparison(df, output_dir=FIGURES_DIR):
    """
    Grouped bar chart comparing all classifiers at the 500-gene level
    across multiple metrics (Accuracy, AUC, Precision, Recall, F1).
    500 genes chosen as representative of the plateau region.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    subset = df[df["gene_count"] == 500].copy()
    if subset.empty:
        logger.warning("No results for 500-gene dataset, skipping comparison chart.")
        return None

    metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
    classifiers = ["SMO", "RandomForest", "MultilayerPerceptron", "J48"]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(metrics))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, clf in enumerate(classifiers):
        row = subset[subset["Classifier"] == clf]
        if row.empty:
            continue
        values = [row[m].values[0] for m in metrics]
        bars = ax.bar(
            x + offsets[i] * width,
            values,
            width,
            label=CLASSIFIER_LABELS[clf],
            color=CLASSIFIER_COLORS[clf],
            edgecolor="white",
            linewidth=0.8,
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, color="#333333",
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classifier Comparison at 500 Genes", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    filepath = os.path.join(output_dir, "classifier_comparison_500.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)

    logger.info(f"Saved classifier comparison chart: {filepath}")
    return filepath


def plot_performance_heatmap(df, output_dir=FIGURES_DIR):
    """
    Heatmap of accuracy values: classifiers (rows) × gene counts (columns).
    Gives a single-glance overview of all 24 experiments.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    pivot = df.pivot_table(
        index="Classifier", columns="gene_count", values="Accuracy"
    )
    # Reorder rows
    row_order = ["SMO", "RandomForest", "MultilayerPerceptron", "J48"]
    pivot = pivot.reindex([c for c in row_order if c in pivot.index])
    pivot.index = [CLASSIFIER_LABELS.get(c, c) for c in pivot.index]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(pivot.values * 100, cmap="YlGnBu", aspect="auto", vmin=92, vmax=98)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c):,}" for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Number of Genes", fontsize=11)
    ax.set_title("Classification Accuracy (%) Heatmap", fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                text = "\u2014"
                color = "#999999"
            else:
                text = f"{val*100:.1f}"
                color = "white" if val > 0.965 else "#333333"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=10)

    filepath = os.path.join(output_dir, "accuracy_heatmap.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)

    logger.info(f"Saved accuracy heatmap: {filepath}")
    return filepath


def generate_all_result_figures(filepath, output_dir=FIGURES_DIR):
    """
    Generate all result analysis figures from the Weka comparison Excel file.

    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    output_dir : str
        Directory to save the figures.

    Returns
    -------
    dict
        Mapping of figure name -> file path.
    """
    df = load_results(filepath)

    figures = {}
    figures["accuracy_vs_genes"] = plot_accuracy_vs_genes(df, output_dir)
    figures["auc_vs_genes"] = plot_auc_vs_genes(df, output_dir)
    figures["classifier_comparison"] = plot_classifier_comparison(df, output_dir)
    figures["accuracy_heatmap"] = plot_performance_heatmap(df, output_dir)

    logger.info(f"Generated {len(figures)} result analysis figures.")
    return figures


# Quick test when run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.results_analyzer <path_to_weka_results.xlsx>")
        print("  e.g. python -m src.results_analyzer weka_results/classifier_comparison.xlsx")
        sys.exit(1)

    filepath = sys.argv[1]

    print("=" * 60)
    print("TCGA-PRAD Results Analyzer")
    print("=" * 60)

    figures = generate_all_result_figures(filepath)

    print("\nGenerated figures:")
    for name, path in figures.items():
        if path:
            size_kb = os.path.getsize(path) / 1024
            print(f"  {name}: {os.path.basename(path)} ({size_kb:.0f} KB)")

    print("\nDone.")