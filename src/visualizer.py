"""
Visualizer Module:
Generates publication-quality figures for the capstone report:
1. PCA plot (tumor vs normal separation)
2. Class distribution bar chart
3. Heatmap of top discriminating genes
4. Gene ranking F-score plot
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from config.settings import (
    COLOR_NORMAL,
    COLOR_TUMOR,
    FIGURE_DPI,
    FIGURES_DIR,
    HEATMAP_TOP_N,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def plot_pca(df, output_dir=FIGURES_DIR):
    """
    PCA scatter plot showing tumor vs normal sample separation
    on the first two principal components.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed expression DataFrame with 'class' column.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(columns=["class"]).values
    y = df["class"].values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, color in [("tumor", COLOR_TUMOR), ("normal", COLOR_NORMAL)]:
        mask = y == label
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color, label=label.capitalize(),
            alpha=0.7, edgecolors="white", linewidth=0.5, s=50
        )

    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)")
    ax.set_title("PCA: Tumor vs Normal Prostate Tissue")
    ax.legend()
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, "pca_tumor_vs_normal.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIGURE_DPI)
    plt.close(fig)

    logger.info(f"Saved PCA plot: {filepath}")
    return filepath


def plot_class_distribution(df, output_dir=FIGURES_DIR):
    """
    Bar chart showing the number of tumor vs normal samples.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled DataFrame with 'class' column.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    counts = df["class"].value_counts()
    colors = [COLOR_TUMOR, COLOR_NORMAL]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Tumor", "Normal"],
        [counts.get("tumor", 0), counts.get("normal", 0)],
        color=colors, edgecolor="white", linewidth=1.5
    )

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 5,
            str(int(height)), ha="center", va="bottom", fontweight="bold"
        )

    ax.set_ylabel("Number of Samples")
    ax.set_title("TCGA-PRAD Sample Distribution")
    ax.set_ylim(0, max(counts) * 1.15)

    filepath = os.path.join(output_dir, "class_distribution.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIGURE_DPI)
    plt.close(fig)

    logger.info(f"Saved class distribution plot: {filepath}")
    return filepath


def plot_heatmap(df, ranking, top_n=HEATMAP_TOP_N, output_dir=FIGURES_DIR):
    """
    Heatmap of the top N discriminating genes across all samples,
    with samples grouped by class.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed expression DataFrame with 'class' column.
    ranking : pd.DataFrame
        Gene ranking from rank_genes().
    top_n : int
        Number of top genes to show.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    top_genes = ranking.head(top_n)["gene"].tolist()

    # Sort samples: normal first, then tumor (for visual grouping)
    df_sorted = df.sort_values("class", ascending=True)
    heatmap_data = df_sorted[top_genes]

    # Build class color bar
    class_colors = df_sorted["class"].map(
        {"tumor": COLOR_TUMOR, "normal": COLOR_NORMAL}
    )

    g = sns.clustermap(
        heatmap_data.T,
        col_cluster=False,  # keep samples in class order
        row_cluster=True,   # cluster genes by similarity
        col_colors=class_colors.values,
        cmap="RdBu_r",
        figsize=(12, 10),
        xticklabels=False,
        yticklabels=True if top_n <= 50 else False,
        cbar_kws={"label": "log2(norm_count+1)"},
    )
    g.fig.suptitle(f"Top {top_n} Discriminating Genes", y=1.02, fontsize=14)

    filepath = os.path.join(output_dir, f"heatmap_top{top_n}.png")
    g.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close("all")

    logger.info(f"Saved heatmap: {filepath}")
    return filepath


def plot_gene_ranking(ranking, top_n=30, output_dir=FIGURES_DIR):
    """
    Horizontal bar chart of the top N genes by F-score.

    Parameters
    ----------
    ranking : pd.DataFrame
        Gene ranking with 'gene' and 'f_score' columns.
    top_n : int
        Number of top genes to display.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    top = ranking.head(top_n).iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    ax.barh(
        top["gene"], top["f_score"],
        color=COLOR_TUMOR, edgecolor="white", linewidth=0.5
    )
    ax.set_xlabel("ANOVA F-Score")
    ax.set_title(f"Top {top_n} Discriminating Genes (ANOVA F-test)")
    ax.grid(True, axis="x", alpha=0.3)

    filepath = os.path.join(output_dir, "gene_ranking_fscore.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIGURE_DPI)
    plt.close(fig)

    logger.info(f"Saved gene ranking plot: {filepath}")
    return filepath


def generate_all_figures(df, ranking, output_dir=FIGURES_DIR):
    """
    Generate all report figures.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed expression DataFrame with 'class' column.
    ranking : pd.DataFrame
        Gene ranking from rank_genes().
    output_dir : str
        Directory to save figures.

    Returns
    -------
    dict
        Mapping of figure name -> file path.
    """
    logger.info("Generating all figures...")

    figures = {
        "pca": plot_pca(df, output_dir),
        "class_dist": plot_class_distribution(df, output_dir),
        "heatmap": plot_heatmap(df, ranking, output_dir=output_dir),
        "gene_ranking": plot_gene_ranking(ranking, output_dir=output_dir),
    }

    logger.info(f"All {len(figures)} figures saved to {output_dir}")
    return figures


# Quick test when run directly
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.downloader import get_labeled_data
    from src.preprocessor import preprocess
    from src.feature_selector import rank_genes

    print("=" * 60)
    print("TCGA-PRAD Visualizer - Test Run")
    print("=" * 60)

    raw_df = get_labeled_data()
    clean_df = preprocess(raw_df)
    ranking = rank_genes(clean_df)

    figures = generate_all_figures(clean_df, ranking)

    print("\nGenerated figures:")
    for name, path in figures.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"  {name}: {path} ({size_kb:.0f} KB)")

    print("\nDone.")