"""
Publication-Grade Visualization Module for Lab 4.
Generates charts for perplexity comparison across N-gram orders & smoothing methods.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ppl_comparison_bar(results_csv: str, output_dir: str = "lab4/results/plots") -> str:
    """
    Generate grouped bar chart comparing Test Set Perplexity across N-gram orders and smoothing methods.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    
    # Filter out INF values for clean plotting (replace INF with NaN or clip for display)
    df_clean = df.copy()
    df_clean['Perplexity_Plot'] = df_clean['Perplexity'].apply(lambda x: np.nan if np.isinf(x) or x > 100000 else x)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Palette
    palette = sns.color_palette("muted", n_colors=4)

    chart = sns.barplot(
        data=df_clean,
        x="Smoothing_Method",
        y="Perplexity_Plot",
        hue="Order",
        palette=palette,
        ax=ax
    )

    ax.set_yscale("log")
    ax.set_title("Test Set Perplexity Comparison across N-gram Orders & Smoothing Methods (Log Scale)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Smoothing Method", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Perplexity (Log Scale)", fontsize=12, fontweight="bold", labelpad=10)
    ax.legend(title="N-gram Model", title_fontsize="11", fontsize="10", loc="upper right")

    # Annotate bars with values
    for p in chart.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:0.1f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=8, rotation=0, xytext=(0, 3),
                textcoords='offset points'
            )

    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "ppl_comparison_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved bar chart plot to '{out_path}'")
    return out_path


def plot_ppl_heatmap(matrix_csv: str, output_dir: str = "lab4/results/plots") -> str:
    """
    Generate heatmap of Test Set Perplexities across Smoothing Methods vs N-gram Orders.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(matrix_csv)
    
    df.set_index("Smoothing Method", inplace=True)
    
    # Convert string INF to np.nan or high numeric value for heatmap representation
    num_df = df.copy()
    for col in num_df.columns:
        num_df[col] = pd.to_numeric(num_df[col].replace("INF", np.nan), errors='coerce')

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(9, 6))

    # Log transform for smooth colormap
    log_matrix = np.log10(num_df)

    sns.heatmap(
        log_matrix,
        annot=df.values,
        fmt="",
        cmap="YlGnBu_r",
        cbar_kws={'label': 'log10(Perplexity)'},
        linewidths=1,
        linecolor="white",
        ax=ax
    )

    ax.set_title("Perplexity Heatmap (Smoothing Method vs. N-gram Order)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Smoothing Method", fontsize=11, fontweight="bold")
    ax.set_xlabel("N-gram Order", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "ppl_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved heatmap plot to '{out_path}'")
    return out_path


def plot_add_k_tuning(add_k_csv: str, output_dir: str = "lab4/results/plots") -> str:
    """
    Generate line plot showing Dev Set Perplexity vs Add-K Parameter K (0 < K < 1).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(add_k_csv)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.lineplot(
        data=df,
        x="K",
        y="Dev_Perplexity",
        hue="Order",
        marker="o",
        linewidth=2.5,
        ax=ax
    )

    ax.set_title("Add-K Parameter Tuning (K vs Dev Set Perplexity)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("K Parameter (0 < K < 1)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dev Set Perplexity (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_yscale("log")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "add_k_tuning.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Add-K tuning plot to '{out_path}'")
    return out_path


def generate_all_plots(results_dir: str = "lab4/results", plots_dir: str = "lab4/results/plots"):
    """
    Generate all visualization figures for Lab 4.
    """
    results_csv = os.path.join(results_dir, "ppl_comparison.csv")
    matrix_csv = os.path.join(results_dir, "ppl_matrix.csv")
    add_k_csv = os.path.join(results_dir, "add_k_dev_tuning.csv")

    if os.path.exists(results_csv):
        plot_ppl_comparison_bar(results_csv, plots_dir)
    if os.path.exists(matrix_csv):
        plot_ppl_heatmap(matrix_csv, plots_dir)
    if os.path.exists(add_k_csv):
        plot_add_k_tuning(add_k_csv, plots_dir)


if __name__ == '__main__':
    generate_all_plots()
