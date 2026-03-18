"""
Data Downloader Module:
Downloads TCGA-PRAD RNA-Seq data from UCSC Xena TCGA Hub,
loads it into a pandas DataFrame, and classifies samples as
tumor or normal based on TCGA barcode conventions.

Dataset: TCGA.PRAD.sampleMap/HiSeqV2
Unit: log2(RSEM normalized count + 1)
Format: genes (Hugo symbols) as rows × samples as columns
        We transpose to samples (rows) × genes (columns) for ML use.
"""

import logging
import os

import pandas as pd
import requests

from config.settings import (
    EXPRESSION_FILENAME,
    EXPRESSION_URL,
    NORMAL_CODES,
    PROJECT_ROOT,
    TUMOR_CODES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default save location: capstone_project/data/
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Download the gzipped TSV from Xenabrowser if not already present
def download_data(url=EXPRESSION_URL, dest_dir=DATA_DIR,
                  filename=EXPRESSION_FILENAME, force=False):
    """
    Parameters
    ----------
    url : str
        URL to the TCGA-PRAD htseq_fpkm.tsv.gz file.
    dest_dir : str
        Directory to save the downloaded file.
    filename : str
        Name for the saved file.
    force : bool
        If True, re-download even if file exists.

    Returns
    -------
    str
        Full path to the downloaded file.
    """
    filepath = os.path.join(dest_dir, filename)

    if os.path.exists(filepath) and not force:
        logger.info(f"Data file already exists: {filepath}")
        return filepath

    logger.info("Downloading TCGA-PRAD data from Xenabrowser...")
    logger.info(f"URL: {url}")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    # Write in chunks to handle large file
    total_bytes = 0
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            total_bytes += len(chunk)

    size_mb = total_bytes / (1024 * 1024)
    logger.info(f"Download complete: {filepath} ({size_mb:.1f} MB)")
    return filepath

# Load the gzipped TSV into a pandas DataFrame and transpose
def load_expression_data(filepath=None):
    """
    Load the gzipped TSV into a pandas DataFrame and transpose so that
    rows = samples and columns = genes.

    The raw file has genes as rows and samples as columns.
    The first column is 'Ensembl_ID' (gene identifiers).

    Parameters
    ----------
    filepath : str or None
        Path to the .tsv.gz file. Defaults to DATA_DIR/EXPRESSION_FILENAME.

    Returns
    -------
    pd.DataFrame
        Transposed DataFrame with samples as rows, genes as columns.
        Index = sample barcodes, columns = Ensembl gene IDs.
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, EXPRESSION_FILENAME)

    logger.info(f"Loading expression data from: {filepath}")

    df = pd.read_csv(filepath, sep="\t", index_col=0)
    logger.info(f"Raw data shape (genes x samples): {df.shape}")

    # Transpose: samples as rows, genes as columns
    df = df.T
    logger.info(f"Transposed shape (samples x genes): {df.shape}")

    return df


# Classify a single TCGA barcode as 'tumor' or 'normal'
def classify_sample(barcode):
    """
    Classify a single TCGA barcode as 'tumor' or 'normal'.

    TCGA barcodes follow the format: TCGA-XX-XXXX-SSV-...
    Positions 13-14 (the SS part after the third hyphen) encode the
    sample type:
        01-09 = Tumor
        10-19 = Normal (adjacent tissue)

    Parameters
    ----------
    barcode : str
        Full TCGA sample barcode, e.g. 'TCGA-2A-A8VL-01A-11R-A37L-07'

    Returns
    -------
    str or None
        'tumor', 'normal', or None if the code is unrecognized.
    """
    try:
        # Split on hyphens and get the 4th segment (index 3)
        # e.g. TCGA-2A-A8VL-01A -> segment '01A'
        parts = barcode.split("-")
        sample_segment = parts[3]           # e.g. '01A'
        sample_code = int(sample_segment[:2])  # e.g. 1

        if sample_code in TUMOR_CODES:
            return "tumor"
        elif sample_code in NORMAL_CODES:
            return "normal"
        else:
            logger.warning(f"Unknown sample code {sample_code} in: {barcode}")
            return None
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not parse barcode '{barcode}': {e}")
        return None

# Add a 'class' column to the DataFrame by classifying each sample barcode
def add_labels(df):
    """
    Add a 'class' column to the DataFrame by classifying each sample
    barcode, then drop any samples with unrecognized barcodes.

    Parameters
    ----------
    df : pd.DataFrame
        Expression DataFrame with TCGA barcodes as the index.

    Returns
    -------
    pd.DataFrame
        DataFrame with a 'class' column ('tumor' or 'normal'),
        unrecognized samples removed.
    """
    logger.info("Classifying samples from TCGA barcodes...")

    df = df.copy()
    df["class"] = df.index.map(classify_sample)

    # Report classification results before filtering
    counts = df["class"].value_counts(dropna=False)
    logger.info(f"Sample classification:\n{counts.to_string()}")

    # Drop unrecognized samples
    n_before = len(df)
    df = df.dropna(subset=["class"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} samples with unrecognized barcodes.")

    logger.info(
        f"Final dataset: {len(df)} samples "
        f"({(df['class'] == 'tumor').sum()} tumor, "
        f"{(df['class'] == 'normal').sum()} normal)"
    )

    return df


# Convenience function: download, load, classify, and return labeled expression df
def get_labeled_data(force_download=False):
    """
    Convenience function: download (if needed), load, classify, and return
    the labeled expression DataFrame ready for preprocessing.

    Parameters
    ----------
    force_download : bool
        If True, re-download the data even if the file exists locally.

    Returns
    -------
    pd.DataFrame
        Samples x genes DataFrame with a 'class' column.
    """
    filepath = download_data(force=force_download)
    df = load_expression_data(filepath)
    df = add_labels(df)
    return df


# Quick test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("TCGA-PRAD Downloader — Test Run")
    print("=" * 60)

    df = get_labeled_data()

    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df["class"].value_counts())
    print(f"\nFirst 5 sample barcodes: {df.index[:5].tolist()}")
    print(f"First 5 gene columns: {df.columns[:5].tolist()}")
    print(f"\nSample values (top-left 3x3):")
    print(df.iloc[:3, :3])
    print("\n Downloader test complete.")