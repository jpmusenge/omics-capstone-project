"""
Preprocessor Module:
Cleans, filters, and prepares the labeled expression DataFrame
for feature selection and Weka export.

Steps:
1. Validate data types (ensure all numeric)
2. Remove duplicate genes (if any)
3. Filter low-expression genes (mean < threshold)
4. Filter low-variance genes (bottom percentile)
5. Sanitize gene names for Weka compatibility
"""

import logging
import re

import numpy as np
import pandas as pd

from config.settings import (
    MIN_MEAN_EXPRESSION,
    VARIANCE_PERCENTILE_CUTOFF,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_data(df):
    """
    Ensure all columns (except 'class') are numeric. Convert if needed,
    coercing errors to NaN, then drop any rows/columns that are entirely NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled expression DataFrame (samples x genes + 'class' column).

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with all expression values as float.
    """
    logger.info("Validating data types...")

    # Separate class column
    labels = df["class"]
    expr = df.drop(columns=["class"])

    # Convert to numeric
    expr = expr.apply(pd.to_numeric, errors="coerce")

    # Check for NaNs introduced by coercion
    nan_count = expr.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} non-numeric values (coerced to NaN).")
        # Drop genes that are entirely NaN
        expr = expr.dropna(axis=1, how="all")
        # Drop samples that are entirely NaN
        expr = expr.dropna(axis=0, how="all")
        # Fill remaining NaN with 0 (conservative)
        expr = expr.fillna(0.0)
    else:
        logger.info("All expression values are numeric. No issues found.")

    df = pd.concat([expr, labels], axis=1)
    return df


def remove_duplicate_genes(df):
    """
    Remove duplicate gene columns, keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Expression DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with unique gene columns.
    """
    # Exclude 'class' from duplicate check
    gene_cols = [c for c in df.columns if c != "class"]
    dupes = pd.Series(gene_cols).duplicated()
    n_dupes = dupes.sum()

    if n_dupes > 0:
        logger.warning(f"Removing {n_dupes} duplicate gene columns.")
        # Keep class + non-duplicate gene columns
        keep_genes = pd.Series(gene_cols)[~dupes].tolist()
        df = df[keep_genes + ["class"]]
    else:
        logger.info("No duplicate genes found.")

    return df


def filter_low_expression(df, threshold=MIN_MEAN_EXPRESSION):
    """
    Remove genes with mean expression below the threshold.
    Since data is log2(norm_count+1), a mean < 1.0 corresponds to
    very low expression across samples.

    Parameters
    ----------
    df : pd.DataFrame
        Expression DataFrame with 'class' column.
    threshold : float
        Minimum mean expression to keep a gene.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    expr = df.drop(columns=["class"])
    gene_means = expr.mean()

    keep = gene_means >= threshold
    n_removed = (~keep).sum()
    n_kept = keep.sum()

    logger.info(
        f"Low-expression filter (mean < {threshold}): "
        f"removed {n_removed}, kept {n_kept}"
    )

    df = pd.concat([expr.loc[:, keep], df["class"]], axis=1)
    return df


def filter_low_variance(df, percentile=VARIANCE_PERCENTILE_CUTOFF):
    """
    Remove genes in the bottom percentile of variance.
    Low-variance genes carry little discriminating signal.

    Parameters
    ----------
    df : pd.DataFrame
        Expression DataFrame with 'class' column.
    percentile : int
        Bottom percentile to remove (e.g. 25 removes bottom 25%).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    expr = df.drop(columns=["class"])
    gene_vars = expr.var()

    cutoff = np.percentile(gene_vars, percentile)
    keep = gene_vars >= cutoff
    n_removed = (~keep).sum()
    n_kept = keep.sum()

    logger.info(
        f"Low-variance filter (bottom {percentile}th pct, cutoff={cutoff:.4f}): "
        f"removed {n_removed}, kept {n_kept}"
    )

    df = pd.concat([expr.loc[:, keep], df["class"]], axis=1)
    return df


def sanitize_gene_names(df):
    """
    Make gene names Weka-compatible by replacing special characters.
    Weka ARFF/CSV has issues with characters like ?, |, and spaces.

    Replaces any character that is not alphanumeric, underscore,
    or period with an underscore. Also removes leading/trailing
    underscores and collapses multiple underscores.

    Parameters
    ----------
    df : pd.DataFrame
        Expression DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with sanitized column names.
    """
    def clean_name(name):
        if name == "class":
            return name
        # Replace non-alphanumeric (except _ and .) with underscore
        cleaned = re.sub(r"[^A-Za-z0-9_.]", "_", name)
        # Collapse multiple underscores
        cleaned = re.sub(r"_+", "_", cleaned)
        # Strip leading/trailing underscores
        cleaned = cleaned.strip("_")
        return cleaned

    original_cols = df.columns.tolist()
    new_cols = [clean_name(c) for c in original_cols]

    # Check for name collisions after cleaning
    if len(set(new_cols)) < len(new_cols):
        # Resolve collisions by appending a suffix
        seen = {}
        resolved = []
        for name in new_cols:
            if name in seen:
                seen[name] += 1
                resolved.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                resolved.append(name)
        new_cols = resolved

    n_changed = sum(1 for o, n in zip(original_cols, new_cols) if o != n)
    if n_changed > 0:
        logger.info(f"Sanitized {n_changed} gene names for Weka compatibility.")
    else:
        logger.info("All gene names are already Weka-compatible.")

    df.columns = new_cols
    return df


def preprocess(df):
    """
    Run the full preprocessing pipeline:
    1. Validate data types
    2. Remove duplicate genes
    3. Filter low-expression genes
    4. Filter low-variance genes
    5. Sanitize gene names for Weka

    Parameters
    ----------
    df : pd.DataFrame
        Raw labeled expression DataFrame from the downloader.

    Returns
    -------
    pd.DataFrame
        Cleaned and filtered DataFrame ready for feature selection.
    """
    logger.info("=" * 50)
    logger.info("Starting preprocessing pipeline...")
    logger.info(f"Input: {df.shape[0]} samples, {df.shape[1] - 1} genes")
    logger.info("=" * 50)

    df = validate_data(df)
    df = remove_duplicate_genes(df)
    df = filter_low_expression(df)
    df = filter_low_variance(df)
    df = sanitize_gene_names(df)

    n_genes = df.shape[1] - 1  # exclude 'class'
    logger.info("=" * 50)
    logger.info(f"Preprocessing complete: {df.shape[0]} samples, {n_genes} genes")
    logger.info("=" * 50)

    return df


# === Quick test when run directly ===
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.downloader import get_labeled_data

    print("=" * 60)
    print("TCGA-PRAD Preprocessor - Test Run")
    print("=" * 60)

    raw_df = get_labeled_data()
    clean_df = preprocess(raw_df)

    print(f"\nBefore: {raw_df.shape[0]} samples, {raw_df.shape[1] - 1} genes")
    print(f"After:  {clean_df.shape[0]} samples, {clean_df.shape[1] - 1} genes")
    print(f"\nClass distribution:")
    print(clean_df["class"].value_counts())
    print(f"\nFirst 10 gene names: {clean_df.columns[:10].tolist()}")
    print(f"\nSample values (top-left 3x3):")
    print(clean_df.iloc[:3, :3])
    print("\nDone.")