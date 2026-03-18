"""
Configuration for the TCGA-PRAD Capstone Pipeline.
All tunable parameters live here.
"""

import os

# PATH SOURCE
# PROJECT_ROOT points to the capstone_project/ folder. Everything else is relative to that
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
WEKA_DIR = os.path.join(PROJECT_ROOT, "weka_data")
GENE_LIST_DIR = os.path.join(PROJECT_ROOT, "gene_lists")

# Create output directories if they don't exist
for d in [FIGURES_DIR, WEKA_DIR, GENE_LIST_DIR]:
    os.makedirs(d, exist_ok=True)

# DATA SOURCE
# UCSC Xena TCGA Hub — legacy hg19 gene expression RNAseq (IlluminaHiSeq)
# Unit: log2(RSEM norm_count + 1), 20,531 Hugo gene symbols × 550 samples
# Dataset ID: TCGA.PRAD.sampleMap/HiSeqV2
EXPRESSION_URL = (
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TCGA.PRAD.sampleMap%2FHiSeqV2.gz"
)
EXPRESSION_FILENAME = "TCGA-PRAD_HiSeqV2.gz"

# TCGA BARCODE RULES 
# Every TCGA sample has a barcode like: TCGA-XX-XXXX-01A-...
# Characters at positions 13-14 tell us the sample type:
#   01-09 = Tumor tissue
#   10-19 = Normal adjacent tissue
TUMOR_CODES = range(1, 10)
NORMAL_CODES = range(10, 20)

# PREPROCESSING 
# Remove genes in the bottom 25% of variance (they carry no signal)
VARIANCE_PERCENTILE_CUTOFF = 25

# Remove genes with mean expression below this (log2 scale)
# log2(FPKM+1) > 1 means FPKM > 1, a standard activity threshold
MIN_MEAN_EXPRESSION = 1.0

# FEATURE SELECTION 
# Gene counts to export for Weka testing
# You'll run classifiers on each to find where accuracy plateaus
GENE_COUNTS = [2000, 1000, 500, 200, 100, 50]

# VISUALIZATION 
FIGURE_DPI = 150
COLOR_TUMOR = "#e74c3c"
COLOR_NORMAL = "#2ecc71"
HEATMAP_TOP_N = 50