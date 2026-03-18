"""
Exporter Module:
Exports feature-selected DataFrames as Weka-ready CSV files
and gene list text files for STRING/KEGG pathway analysis.

Weka CSV requirements:
- 'class' column must be the LAST column
- No index column (sample barcodes not included)
- Clean column names (handled by preprocessor sanitization)

Gene lists:
- One gene name per line, plain text
- Used for pasting into STRING (string-db.org) or KEGG
"""

import logging
import os

import pandas as pd

from config.settings import GENE_LIST_DIR, WEKA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def export_weka_csv(df, k, output_dir=WEKA_DIR):
    """
    Export a feature-selected DataFrame as a Weka-ready CSV.

    The 'class' column is moved to the last position and
    sample barcodes (index) are excluded.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-selected DataFrame with genes + 'class' column.
    k : int
        Gene count level (used in the filename).
    output_dir : str
        Directory to save the CSV file.

    Returns
    -------
    str
        Path to the exported CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure 'class' is the last column
    gene_cols = [c for c in df.columns if c != "class"]
    df_export = df[gene_cols + ["class"]]

    filename = f"prad_top{k}.csv"
    filepath = os.path.join(output_dir, filename)
    df_export.to_csv(filepath, index=False)

    logger.info(f"Exported Weka CSV: {filepath} "
                f"({df_export.shape[0]} samples, {len(gene_cols)} genes)")
    return filepath


def export_all_weka_csvs(selected_datasets, output_dir=WEKA_DIR):
    """
    Export Weka CSVs for all gene count levels.

    Parameters
    ----------
    selected_datasets : dict
        Mapping of gene count -> DataFrame (from select_at_all_levels).
    output_dir : str
        Directory to save the CSV files.

    Returns
    -------
    dict
        Mapping of gene count -> exported file path.
    """
    paths = {}
    for k, df in selected_datasets.items():
        path = export_weka_csv(df, k, output_dir)
        paths[k] = path
    return paths


def export_gene_list(ranking, k, output_dir=GENE_LIST_DIR):
    """
    Export the top K gene names as a plain text file (one per line).
    Useful for pasting into STRING or KEGG.

    Parameters
    ----------
    ranking : pd.DataFrame
        Gene ranking from rank_genes() with 'gene' column.
    k : int
        Number of top genes to export.
    output_dir : str
        Directory to save the gene list.

    Returns
    -------
    str
        Path to the exported gene list file.
    """
    os.makedirs(output_dir, exist_ok=True)

    top_genes = ranking.head(k)["gene"].tolist()

    filename = f"top{k}_genes.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write("\n".join(top_genes))

    logger.info(f"Exported gene list: {filepath} ({len(top_genes)} genes)")
    return filepath


def export_all_gene_lists(ranking, gene_counts, output_dir=GENE_LIST_DIR):
    """
    Export gene lists at all predefined count levels.

    Parameters
    ----------
    ranking : pd.DataFrame
        Gene ranking from rank_genes().
    gene_counts : list of int
        Gene count levels to export.
    output_dir : str
        Directory to save the gene lists.

    Returns
    -------
    dict
        Mapping of gene count -> exported file path.
    """
    paths = {}
    for k in gene_counts:
        path = export_gene_list(ranking, k, output_dir)
        paths[k] = path
    return paths


def export_ranking(ranking, output_dir=GENE_LIST_DIR):
    """
    Export the full gene ranking as a CSV for reference.

    Parameters
    ----------
    ranking : pd.DataFrame
        Gene ranking with 'gene', 'f_score', 'p_value' columns.
    output_dir : str
        Directory to save the ranking file.

    Returns
    -------
    str
        Path to the exported ranking CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, "full_gene_ranking.csv")
    ranking.to_csv(filepath, index=False)

    logger.info(f"Exported full gene ranking: {filepath} ({len(ranking)} genes)")
    return filepath


# Quick test when run directly
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.downloader import get_labeled_data
    from src.preprocessor import preprocess
    from src.feature_selector import rank_genes, select_at_all_levels
    from config.settings import GENE_COUNTS

    print("=" * 60)
    print("TCGA-PRAD Exporter - Test Run")
    print("=" * 60)

    raw_df = get_labeled_data()
    clean_df = preprocess(raw_df)
    ranking = rank_genes(clean_df)
    selected = select_at_all_levels(clean_df, ranking)

    # Export Weka CSVs
    weka_paths = export_all_weka_csvs(selected)

    # Export gene lists
    gene_list_paths = export_all_gene_lists(ranking, GENE_COUNTS)

    # Export full ranking
    ranking_path = export_ranking(ranking)

    print(f"\nWeka CSV files:")
    for k, path in weka_paths.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.0f} KB)")

    print(f"\nGene list files:")
    for k, path in gene_list_paths.items():
        print(f"  {path}")

    print(f"\nFull ranking: {ranking_path}")
    print("\nDone.")