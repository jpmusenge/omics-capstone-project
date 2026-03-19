"""
TCGA-PRAD ML Classification Pipeline:
Entry point that runs the full pipeline end-to-end:

1. Download TCGA-PRAD expression data from UCSC Xena
2. Preprocess: validate, filter low-expression/low-variance genes, sanitize names
3. Feature selection: ANOVA F-test ranking at multiple gene count levels
4. Export: Weka-ready CSVs and gene lists for STRING/KEGG
5. Visualize: PCA, class distribution, heatmap, gene ranking plots

After running this script, the next steps are manual:
- Load Weka CSVs into Weka GUI and run classifiers
- Paste gene lists into STRING for pathway analysis
"""

import sys
import time

from config.settings import GENE_COUNTS
from src.downloader import get_labeled_data
from src.preprocessor import preprocess
from src.feature_selector import rank_genes, select_at_all_levels
from src.exporter import export_all_weka_csvs, export_all_gene_lists, export_ranking
from src.visualizer import generate_all_figures


def main():
    start = time.time()

    print("=" * 60)
    print("TCGA-PRAD ML Classification Pipeline")
    print("=" * 60)

    # Download and label
    print("\n[1/5] Downloading and labeling data...")
    df = get_labeled_data()

    # Preprocess
    print("\n[2/5] Preprocessing...")
    df_clean = preprocess(df)

    # Feature selection
    print("\n[3/5] Running feature selection (ANOVA F-test)...")
    ranking = rank_genes(df_clean)
    selected = select_at_all_levels(df_clean, ranking)

    # Export
    print("\n[4/5] Exporting Weka CSVs and gene lists...")
    weka_paths = export_all_weka_csvs(selected)
    gene_list_paths = export_all_gene_lists(ranking, GENE_COUNTS)
    ranking_path = export_ranking(ranking)

    # Visualize
    print("\n[5/5] Generating figures...")
    figures = generate_all_figures(df_clean, ranking)

    # Summary
    elapsed = time.time() - start
    n_genes_final = df_clean.shape[1] - 1

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nData:    {df_clean.shape[0]} samples "
          f"({(df_clean['class'] == 'tumor').sum()} tumor, "
          f"{(df_clean['class'] == 'normal').sum()} normal)")
    print(f"Genes:   {df.shape[1] - 1} raw -> {n_genes_final} after filtering")
    print(f"Top gene: {ranking.iloc[0]['gene']} "
          f"(F={ranking.iloc[0]['f_score']:.1f})")

    print(f"\nWeka CSVs ({len(weka_paths)} files):")
    for k in sorted(weka_paths.keys(), reverse=True):
        print(f"  weka_data/prad_top{k}.csv")

    print(f"\nGene lists ({len(gene_list_paths)} files):")
    for k in sorted(gene_list_paths.keys(), reverse=True):
        print(f"  gene_lists/top{k}_genes.txt")
    print(f"  gene_lists/full_gene_ranking.csv")

    print(f"\nFigures ({len(figures)} files):")
    for name in figures:
        print(f"  figures/{name}")

    print(f"\nTime elapsed: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()