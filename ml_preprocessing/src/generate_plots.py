"""
Visualization Generator Module
==============================
Generates publication-quality charts for:
1. Overall class distribution in the integrated cyberbullying dataset
2. Source dataset contribution breakdown
3. Cross-tabulation heatmap of dataset sources vs categories
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
integrated_csv = os.path.join(base_dir, "datasets", "processed", "cyberbullying_integrated_cleaned.csv")
vis_dir = os.path.join(base_dir, "visualizations")
os.makedirs(vis_dir, exist_ok=True)

df = pd.read_csv(integrated_csv)

# 1. Overall Class Distribution
plt.figure(figsize=(10, 5.5))
sns.set_theme(style="whitegrid")

order = ["Not Cyberbullying", "Insult", "Hate Speech", "Harassment", "Threat"]
palette = ["#2e7d32", "#1565c0", "#6a1b9a", "#e65100", "#c62828"]

ax = sns.countplot(data=df, x="label_name", order=order, hue="label_name", palette=palette, legend=False)
total = len(df)

for p in ax.patches:
    h = p.get_height()
    pct = (h / total) * 100
    ax.annotate(
        f"{h:,}\n({pct:.1f}%)",
        xy=(p.get_x() + p.get_width() / 2, h),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

plt.title("Class Distribution in Integrated Cyberbullying Dataset (N = 107,613)", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Target Category", fontsize=11)
plt.ylabel("Record Count", fontsize=11)
plt.ylim(0, max(df["label_name"].value_counts()) * 1.15)
plt.tight_layout()
p1 = os.path.join(vis_dir, "class_distribution_raw.png")
plt.savefig(p1, dpi=300)
plt.close()
print(f"Saved {p1}")

# 2. Source Dataset Contribution Breakdown
plt.figure(figsize=(9, 5.5))
ds_counts = df["original_dataset"].value_counts()
labels_pretty = [
    "Kaggle Cyberbullying Tweets (42.3%)",
    "Jigsaw Toxic Comments (24.1%)",
    "Davidson Hate Speech (22.8%)",
    "TRAC-1 Aggression (10.8%)",
]
colors = ["#1976d2", "#388e3c", "#fbc02d", "#8e24aa"]

plt.pie(
    ds_counts.values,
    labels=labels_pretty,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 11, "fontweight": "medium"},
)
plt.title("Contribution of Public Benchmark Datasets in Unified Corpus", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
p2 = os.path.join(vis_dir, "dataset_contribution.png")
plt.savefig(p2, dpi=300)
plt.close()
print(f"Saved {p2}")

# 3. Cross-Tab Heatmap
plt.figure(figsize=(9, 6))
ct = pd.crosstab(df["original_dataset"], df["label_name"])[order]
ct.index = ["Cyberbullying Tweets", "Davidson Hate Speech", "Jigsaw Toxic", "TRAC-1 Aggression"]

sns.heatmap(ct, annot=True, fmt=",d", cmap="Blues", cbar=True, linewidths=1)
plt.title("Dataset Source vs Target Category Mapping Heatmap", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Target Category", fontsize=11)
plt.ylabel("Original Source Dataset", fontsize=11)
plt.tight_layout()
p3 = os.path.join(vis_dir, "dataset_category_heatmap.png")
plt.savefig(p3, dpi=300)
plt.close()
print(f"Saved {p3}")

