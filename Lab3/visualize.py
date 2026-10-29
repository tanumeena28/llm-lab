"""
Visualization Module for Lab 3.
Generates charts for sentence length distributions, Task 3 data size scaling,
and Task 4 vocabulary size scaling.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'


def plot_length_distributions(data_dir: str = "Lab3/data", output_dir: str = "Lab3/results/plots"):
    """Plots character length distributions for Train, Dev, and Test splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    dev_df = pd.read_csv(os.path.join(data_dir, "dev.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    train_df = pd.read_csv(os.path.join(data_dir, "train_100k.csv"))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    # Histograms normalized to density
    bins = np.linspace(15, 300, 40)
    ax.hist(train_df['char_length'], bins=bins, alpha=0.4, density=True, label='Train Pool (100k+)', color='#1f77b4')
    ax.hist(dev_df['char_length'], bins=bins, alpha=0.5, density=True, label='Dev Set (1,000)', color='#ff7f0e')
    ax.hist(test_df['char_length'], bins=bins, alpha=0.5, density=True, label='Test Set (1,000)', color='#2ca02c')

    ax.set_title("Sentence Length Distribution Across Dataset Splits (IndicCorp v2)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Sentence Character Length", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "sentence_length_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot: '{save_path}'")


def plot_task3_data_size_scaling(results_csv: str = "Lab3/results/task3_data_size_results.csv", output_dir: str = "Lab3/results/plots"):
    """Plots Subword Fertility vs Training Data Size for BPE and WordPiece."""
    if not os.path.exists(results_csv):
        return
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    sizes = ['100k', '300k', '500k', '1m']
    x = np.arange(len(sizes))
    width = 0.35
    
    # Dev Fertility
    bpe_dev = df[df['Algorithm'] == 'BPE']['Dev_Fertility'].values
    wp_dev = df[df['Algorithm'] == 'WordPiece']['Dev_Fertility'].values
    
    ax1.bar(x - width/2, bpe_dev, width, label='BPE', color='#3498db', edgecolor='black', linewidth=0.8)
    ax1.bar(x + width/2, wp_dev, width, label='WordPiece', color='#e74c3c', edgecolor='black', linewidth=0.8)
    ax1.set_title("Dev Set Subword Fertility vs Training Corpus Size", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Training Data Size (Sentences)", fontsize=11)
    ax1.set_ylabel("Subword Fertility (Tokens / Word)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['100K', '300K', '500K', '1M'])
    ax1.set_ylim(0, max(max(bpe_dev), max(wp_dev)) * 1.25)
    ax1.legend(fontsize=10)
    for i in range(len(sizes)):
        ax1.text(x[i] - width/2, bpe_dev[i] + 0.02, f"{bpe_dev[i]:.2f}", ha='center', fontsize=9)
        ax1.text(x[i] + width/2, wp_dev[i] + 0.02, f"{wp_dev[i]:.2f}", ha='center', fontsize=9)

    # Test Fertility
    bpe_test = df[df['Algorithm'] == 'BPE']['Test_Fertility'].values
    wp_test = df[df['Algorithm'] == 'WordPiece']['Test_Fertility'].values
    
    ax2.bar(x - width/2, bpe_test, width, label='BPE', color='#2ecc71', edgecolor='black', linewidth=0.8)
    ax2.bar(x + width/2, wp_test, width, label='WordPiece', color='#9b59b6', edgecolor='black', linewidth=0.8)
    ax2.set_title("Test Set Subword Fertility vs Training Corpus Size", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Training Data Size (Sentences)", fontsize=11)
    ax2.set_ylabel("Subword Fertility (Tokens / Word)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['100K', '300K', '500K', '1M'])
    ax2.set_ylim(0, max(max(bpe_test), max(wp_test)) * 1.25)
    ax2.legend(fontsize=10)
    for i in range(len(sizes)):
        ax2.text(x[i] - width/2, bpe_test[i] + 0.02, f"{bpe_test[i]:.2f}", ha='center', fontsize=9)
        ax2.text(x[i] + width/2, wp_test[i] + 0.02, f"{wp_test[i]:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "task3_training_size_scaling.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot: '{save_path}'")


def plot_task4_vocab_size_scaling(results_csv: str = "Lab3/results/task4_vocab_size_results.csv", output_dir: str = "Lab3/results/plots"):
    """Plots Subword Fertility vs Vocabulary Size for BPE and WordPiece."""
    if not os.path.exists(results_csv):
        return
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    vocabs = ['20k', '30k', '50k']
    x = np.arange(len(vocabs))
    width = 0.35
    
    bpe_dev = df[df['Algorithm'] == 'BPE']['Dev_Fertility'].values
    wp_dev = df[df['Algorithm'] == 'WordPiece']['Dev_Fertility'].values
    
    ax1.bar(x - width/2, bpe_dev, width, label='BPE', color='#34495e', edgecolor='black', linewidth=0.8)
    ax1.bar(x + width/2, wp_dev, width, label='WordPiece', color='#f39c12', edgecolor='black', linewidth=0.8)
    ax1.set_title("Dev Set Subword Fertility vs Vocabulary Size", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Vocabulary Size", fontsize=11)
    ax1.set_ylabel("Subword Fertility (Tokens / Word)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['20K', '30K', '50K'])
    ax1.set_ylim(0, max(max(bpe_dev), max(wp_dev)) * 1.25)
    ax1.legend(fontsize=10)
    for i in range(len(vocabs)):
        ax1.text(x[i] - width/2, bpe_dev[i] + 0.02, f"{bpe_dev[i]:.2f}", ha='center', fontsize=9)
        ax1.text(x[i] + width/2, wp_dev[i] + 0.02, f"{wp_dev[i]:.2f}", ha='center', fontsize=9)

    bpe_test = df[df['Algorithm'] == 'BPE']['Test_Fertility'].values
    wp_test = df[df['Algorithm'] == 'WordPiece']['Test_Fertility'].values
    
    ax2.bar(x - width/2, bpe_test, width, label='BPE', color='#16a085', edgecolor='black', linewidth=0.8)
    ax2.bar(x + width/2, wp_test, width, label='WordPiece', color='#d35400', edgecolor='black', linewidth=0.8)
    ax2.set_title("Test Set Subword Fertility vs Vocabulary Size", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Vocabulary Size", fontsize=11)
    ax2.set_ylabel("Subword Fertility (Tokens / Word)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['20K', '30K', '50K'])
    ax2.set_ylim(0, max(max(bpe_test), max(wp_test)) * 1.25)
    ax2.legend(fontsize=10)
    for i in range(len(vocabs)):
        ax2.text(x[i] - width/2, bpe_test[i] + 0.02, f"{bpe_test[i]:.2f}", ha='center', fontsize=9)
        ax2.text(x[i] + width/2, wp_test[i] + 0.02, f"{wp_test[i]:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "task4_vocab_size_scaling.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot: '{save_path}'")


if __name__ == '__main__':
    plot_length_distributions()
    plot_task3_data_size_scaling()
    plot_task4_vocab_size_scaling()
