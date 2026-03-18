"""
Feature Selector Module:
Ranks genes by their ability to discriminate tumor from normal using ANOVA F-test, then selects the top K genes at each predefined gene count level.

The ANOVA F-test measures how different the means of each gene are between the two classes (tumor vs normal). Higher F-score = more discriminating gene.
"""

import logging

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from config.settings import GENE_COUNTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def rank_genes(df):
    """
    Rank all genes by ANOVA F-score (tumor vs normal).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with samples as rows, genes as columns,
        and a 'class' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['gene', 'f_score', 'p_value'],
        sorted by f_score descending.
    """
    logger.info("Ranking genes by ANOVA F-test...")

    X = df.drop(columns=["class"])
    y = df["class"]

    f_scores, p_values = f_classif(X, y)

    ranking = pd.DataFrame({
        "gene": X.columns,
        "f_score": f_scores,
        "p_value": p_values,
    })
    ranking = ranking.sort_values("f_score", ascending=False).reset_index(drop=True)

    logger.info(f"Ranked {len(ranking)} genes. "
                f"Top gene: {ranking.iloc[0]['gene']} "
                f"(F={ranking.iloc[0]['f_score']:.2f}, "
                f"p={ranking.iloc[0]['p_value']:.2e})")

    return ranking


def select_top_genes(df, ranking, k):
    """
    Select the top K genes from the ranked list and return a
    filtered DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed expression DataFrame with 'class' column.
    ranking : pd.DataFrame
        Gene ranking from rank_genes().
    k : int
        Number of top genes to select.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only the top K genes + 'class'.
    """
    top_genes = ranking.head(k)["gene"].tolist()
    selected = df[top_genes + ["class"]]
    return selected


def select_at_all_levels(df, ranking, gene_counts=GENE_COUNTS):
    """
    Select top genes at each predefined count level.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed expression DataFrame with 'class' column.
    ranking : pd.DataFrame
        Gene ranking from rank_genes().
    gene_counts : list of int
        Gene count levels to select (e.g. [2000, 1000, 500, 200, 100, 50]).

    Returns
    -------
    dict
        Mapping of gene count -> filtered DataFrame.
        e.g. {2000: df_top2000, 1000: df_top1000, ...}
    """
    n_available = len(ranking)
    results = {}

    for k in gene_counts:
        if k > n_available:
            logger.warning(
                f"Requested {k} genes but only {n_available} available. "
                f"Using all {n_available}."
            )
            actual_k = n_available
        else:
            actual_k = k

        selected = select_top_genes(df, ranking, actual_k)
        n_genes = selected.shape[1] - 1  # exclude 'class'
        results[k] = selected
        logger.info(f"Top {k}: selected {n_genes} genes")

    return results


# Quick test when run directly 
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.downloader import get_labeled_data
    from src.preprocessor import preprocess

    print("=" * 60)
    print("TCGA-PRAD Feature Selector - Test Run")
    print("=" * 60)

    raw_df = get_labeled_data()
    clean_df = preprocess(raw_df)

    ranking = rank_genes(clean_df)
    results = select_at_all_levels(clean_df, ranking)

    print(f"\nTop 10 discriminating genes:")
    print(ranking.head(10).to_string(index=False))

    print(f"\nDatasets generated:")
    for k, sel_df in results.items():
        print(f"  Top {k}: {sel_df.shape[0]} samples x {sel_df.shape[1] - 1} genes")

    print("\nDone.")