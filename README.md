# Comparing ML Classifiers for Prostate Adenocarcinoma Classification from RNA-Seq Data

**Identifying Minimum Gene Signatures and Their Biological Relevance**

> OmicsLogic — AI and Machine Learning for Omics Data Analysis | Capstone Project | April 2026

---

## Overview

This project investigates whether machine learning algorithms can accurately classify prostate adenocarcinoma tissue from normal tissue using RNA-Seq gene expression data, which genes are most discriminating, and how compact the gene signature can be while maintaining classification accuracy.

Using **550 TCGA-PRAD samples** (498 tumor, 52 normal) and **20,530 genes**, the pipeline filters, ranks, and exports gene subsets for classification benchmarking in Weka across four major algorithmic families.

### Key Findings

| Metric | Best Result | Algorithm |
|--------|------------|-----------|
| Highest Accuracy | **97.3%** | SMO (SVM) |
| Highest AUC | **0.976** | Random Forest |
| Accuracy Plateau | **100–500 genes** | All classifiers |
| Top Gene | **SIM2** (F=345.4) | — |

The top discriminating genes (SIM2, HOXC6, DLX1, HPN) are established prostate cancer biomarkers already in clinical diagnostic panels (SelectMDx), providing strong biological validation.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TCGA-PRAD RNA-Seq Data                    │
│              UCSC Xena Hub · 550 samples · 20,530 genes     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  1. DOWNLOAD (src/downloader.py)                            │
│     • Fetch HiSeqV2.gz from Xena TCGA Hub                  │
│     • Transpose: genes×samples → samples×genes              │
│     • Classify samples via TCGA barcode (498T / 52N)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. PREPROCESS (src/preprocessor.py)                        │
│     • Remove low-expression genes (mean < 1.0) → -3,461    │
│     • Remove low-variance genes (bottom 25%) → -4,267       │
│     • Sanitize gene names for Weka                          │
│     • Result: 12,802 informative genes                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. FEATURE SELECTION (src/feature_selector.py)             │
│     • ANOVA F-test ranking (tumor vs normal)                │
│     • Top gene subsets: 2000 / 1000 / 500 / 200 / 100 / 50 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────┐  ┌───────────────────────────┐
│  4. EXPORT (src/exporter.py) │  │ 5. VISUALIZE              │
│     • Weka-ready CSVs        │  │    (src/visualizer.py)    │
│     • Gene lists for STRING  │  │    • PCA plot             │
│     • Full ranking CSV       │  │    • Class distribution   │
└──────────────┬───────────────┘  │    • Heatmap (top 50)     │
               │                  │    • F-score ranking      │
               ▼                  └───────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  6. WEKA CLASSIFICATION (manual)                            │
│     • Random Forest    (ensemble)     → 96.5% acc, 0.976 AUC│
│     • SMO / SVM        (kernel)       → 97.3% acc, 0.925 AUC│
│     • MultilayerPerceptron (neural)   → 97.1% acc, 0.972 AUC│
│     • J48 Decision Tree (tree)        → 95.5% acc, 0.857 AUC│
│     • 10-fold cross-validation at each gene count threshold │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  7. BIOLOGICAL INTERPRETATION                               │
│     • STRING protein interaction network                    │
│     • GO / KEGG pathway enrichment                          │
│     • PubMed validation of top genes                        │
│     • Top genes match SelectMDx clinical panel              │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
capstone_project/
├── main.py                     ← Run full pipeline end-to-end
├── config/
│   ├── __init__.py
│   └── settings.py             ← All configurable parameters
├── src/
│   ├── __init__.py
│   ├── downloader.py           ← TCGA data acquisition
│   ├── preprocessor.py         ← Filtering & cleaning
│   ├── feature_selector.py     ← ANOVA F-test gene ranking
│   ├── exporter.py             ← Weka CSV & gene list export
│   └── visualizer.py           ← Report figures
├── data/                       ← Downloaded expression data (gitignored)
├── figures/                    ← Generated plots
│   ├── pca_tumor_vs_normal.png
│   ├── class_distribution.png
│   ├── heatmap_top50.png
│   └── gene_ranking_fscore.png
├── weka_data/                  ← Weka-ready CSVs
│   ├── prad_top50.csv
│   ├── prad_top100.csv
│   ├── prad_top200.csv
│   ├── prad_top500.csv
│   ├── prad_top1000.csv
│   └── prad_top2000.csv
├── gene_lists/                 ← Gene lists for STRING/KEGG
│   ├── top50_genes.txt ... top2000_genes.txt
│   └── full_gene_ranking.csv
├── requirements.txt
└── .gitignore
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/jpmusenge/omics-capstone-project.git
cd omics-capstone-project

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

The pipeline will:
1. Download TCGA-PRAD data (~28 MB) from UCSC Xena
2. Preprocess and filter genes (20,530 → 12,802)
3. Rank genes by ANOVA F-test
4. Export Weka-ready CSVs at 6 gene count thresholds
5. Generate 4 report figures

After the pipeline completes, load the CSV files into [Weka](https://www.cs.waikato.ac.nz/ml/weka/) to run classification.

---

## Results at a Glance

### Classification Accuracy Across Algorithms

| Genes | SMO (SVM) | Random Forest | MLP (ANN) | J48 |
|------:|----------:|--------------:|----------:|----:|
| 50 | 96.36% | 96.55% | 96.36% | 94.36% |
| 100 | 96.91% | 96.36% | 96.36% | 94.36% |
| 200 | 95.64% | 96.18% | 96.36% | 93.64% |
| 500 | 96.55% | 95.64% | **97.09%** | 94.36% |
| 1,000 | 97.09% | 96.00% | **97.09%** | 95.45% |
| 2,000 | **97.27%** | 95.45% | — | 94.91% |

### Top 5 Discriminating Genes

| Rank | Gene | F-Score | Known Prostate Cancer Role |
|------|------|---------|---------------------------|
| 1 | SIM2 | 345.4 | Biomarker & immunotherapy target |
| 2 | EPHA10 | 300.9 | Receptor tyrosine kinase |
| 3 | HOXC6 | 294.3 | In SelectMDx clinical diagnostic panel |
| 4 | HPN | 275.7 | Serine protease, overexpressed in PCa |
| 5 | PYCR1 | 271.1 | Metabolic enzyme, cancer cell proliferation |

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python 3.12** | Pipeline development |
| **pandas / NumPy** | Data manipulation |
| **scikit-learn** | ANOVA F-test feature selection, PCA |
| **matplotlib / seaborn** | Visualization |
| **Weka 3.8** | ML classification (RF, SVM, ANN, J48) |
| **STRING** | Protein interaction & pathway analysis |
| **TCGA / UCSC Xena** | Data source |

---

## Dataset

- **Source**: [UCSC Xena TCGA Hub](https://xenabrowser.net/datapages/?cohort=TCGA%20Prostate%20Cancer%20(PRAD))
- **Dataset ID**: TCGA.PRAD.sampleMap/HiSeqV2
- **Samples**: 550 (498 tumor, 52 normal adjacent tissue)
- **Genes**: 20,530 (Hugo symbols)
- **Unit**: log2(RSEM normalized count + 1)
- **Platform**: Illumina HiSeq 2000

---

## References

1. Arredouani et al. (2009) — SIM2 as biomarker and immunotherapy target in prostate cancer. *Clinical Cancer Research*
2. Cancer Genome Atlas Research Network (2015) — Molecular taxonomy of primary prostate cancer. *Cell*
3. Van Neste et al. (2016) — HOXC6/DLX1 urinary biomarker panel for prostate cancer. *European Urology*
4. Lancashire et al. (2010) — ANN gene expression profiling in breast cancer. *Breast Cancer Research and Treatment*
5. Kourou et al. (2015) — ML applications in cancer prognosis and prediction. *CSBJ*

---

## Author

**Joseph Musenge** — CS Major, Software Engineering & ML  
OmicsLogic Capstone Project · April 2026

---
