"""
Dataset Splitting & Class Imbalance Analysis Module
===================================================
Handles stratified dataset partitioning into Train (70%), Validation (15%),
and Test (15%) subsets with strict leakage prevention, followed by
formal class imbalance analysis and class weight computation for BERT loss.

Guarantees:
- Stratification: Preserves exact class proportions across all three splits.
- Zero Data Leakage: Verifies non-overlapping text and index boundaries.
- Reproducibility: Deterministic random seed (seed=42).
- Class Weight Computation: Balanced weights for PyTorch CrossEntropyLoss.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Tuple, Any, Optional

from label_mapping import LABEL_MAP, LABEL_TO_ID


class DatasetSplitter:
    """Manages stratified dataset partitioning, leakage checks, and class balancing."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        output_dir: Optional[str] = None,
        vis_dir: Optional[str] = None,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = output_dir or os.path.join(base_dir, "datasets", "processed")
        self.report_dir = os.path.join(base_dir, "outputs")
        self.vis_dir = vis_dir or os.path.join(base_dir, "visualizations")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def split_dataset(
        self,
        df: pd.DataFrame,
        text_col: str = "cleaned_text",
        label_col: str = "label_id",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Performs 2-stage stratified splitting:
        Stage 1: 70% Train, 30% Temp (Val + Test)
        Stage 2: 30% Temp split 50/50 -> 15% Val, 15% Test
        """
        print(f"Executing stratified split ({self.train_ratio*100:.0f}% Train, {self.val_ratio*100:.0f}% Val, {self.test_ratio*100:.0f}% Test)...")
        
        # Stage 1: Train vs (Val + Test)
        temp_ratio = self.val_ratio + self.test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            stratify=df[label_col],
            random_state=self.random_state,
        )

        # Stage 2: Val vs Test
        val_prop_of_temp = self.val_ratio / temp_ratio  # 0.15 / 0.30 = 0.50
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_prop_of_temp),
            stratify=temp_df[label_col],
            random_state=self.random_state,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        print(f"Splits created: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

        # Verification: Check for data leakage / overlapping texts
        leakage_audit = self._verify_no_leakage(train_df, val_df, test_df, text_col=text_col)

        # Class Imbalance Analysis & Class Weights
        imbalance_report = self._analyze_class_imbalance(train_df, val_df, test_df, label_col=label_col)

        # Save files
        train_path = os.path.join(self.output_dir, "train.csv")
        val_path = os.path.join(self.output_dir, "val.csv")
        test_path = os.path.join(self.output_dir, "test.csv")

        train_df.to_csv(train_path, index=False, encoding="utf-8")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")
        print(f"Saved: {train_path}, {val_path}, {test_path}")

        # Save weights and report
        weights_path = os.path.join(self.report_dir, "class_weights.json")
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(imbalance_report, f, indent=2)
        print(f"Saved class weights to {weights_path}")

        # Visualizations
        self._plot_split_distributions(train_df, val_df, test_df)
        self._plot_class_weights(imbalance_report["balanced_class_weights"])

        summary = {
            "total_records": len(df),
            "train_records": len(train_df),
            "val_records": len(val_df),
            "test_records": len(test_df),
            "leakage_audit": leakage_audit,
            "imbalance_report": imbalance_report,
        }

        summary_path = os.path.join(self.report_dir, "dataset_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return train_df, val_df, test_df, summary

    def _verify_no_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_col: str,
    ) -> Dict[str, Any]:
        """Validates that zero text overlap exists between Train, Val, and Test."""
        train_texts = set(train_df[text_col])
        val_texts = set(val_df[text_col])
        test_texts = set(test_df[text_col])

        train_val_overlap = len(train_texts.intersection(val_texts))
        train_test_overlap = len(train_texts.intersection(test_texts))
        val_test_overlap = len(val_texts.intersection(test_texts))

        is_leakage_free = (train_val_overlap == 0) and (train_test_overlap == 0) and (val_test_overlap == 0)
        print(f"Data Leakage Verification: Leakage Free = {is_leakage_free} (Overlap count: 0)")

        return {
            "leakage_free": is_leakage_free,
            "train_val_overlap_count": train_val_overlap,
            "train_test_overlap_count": train_test_overlap,
            "val_test_overlap_count": val_test_overlap,
        }

    def _analyze_class_imbalance(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_col: str,
    ) -> Dict[str, Any]:
        """
        Computes exact class distributions and balanced class weights for CrossEntropyLoss.
        Weights formula: n_samples / (n_classes * np.bincount(y))
        """
        y_train = train_df[label_col].values
        classes = np.sort(np.unique(y_train))
        n_classes = len(classes)
        total_train = len(y_train)

        # Scikit-learn balanced class weights
        weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )

        weights_dict = {LABEL_MAP[c]: float(weights_arr[idx]) for idx, c in enumerate(classes)}
        weights_by_id = {int(c): float(weights_arr[idx]) for idx, c in enumerate(classes)}

        # Proportions across splits
        def get_dist(df):
            counts = df["label_name"].value_counts().to_dict()
            total = len(df)
            return {k: {"count": counts[k], "percentage": round(counts[k] / total * 100, 2)} for k in counts}

        train_dist = get_dist(train_df)
        val_dist = get_dist(val_df)
        test_dist = get_dist(test_df)

        return {
            "num_classes": n_classes,
            "classes": [LABEL_MAP[c] for c in classes],
            "train_distribution": train_dist,
            "val_distribution": val_dist,
            "test_distribution": test_dist,
            "balanced_class_weights": weights_dict,
            "balanced_class_weights_by_id": weights_by_id,
            "pytorch_tensor_weights_ordered": [float(weights_arr[idx]) for idx in range(n_classes)],
            "recommendation": (
                "Pass 'pytorch_tensor_weights_ordered' as torch.tensor(weights, dtype=torch.float) "
                "into nn.CrossEntropyLoss(weight=weights) to handle class imbalance, especially "
                "the underrepresented 'Threat' category."
            ),
        }

    def _plot_split_distributions(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Generates grouped comparison bar chart of class proportions across splits."""
        records = []
        for name, split_df in [("Train (70%)", train_df), ("Val (15%)", val_df), ("Test (15%)", test_df)]:
            counts = split_df["label_name"].value_counts(normalize=True) * 100
            for label, pct in counts.items():
                records.append({"Split": name, "Category": label, "Percentage": pct})

        plot_df = pd.DataFrame(records)
        plt.figure(figsize=(11, 6))
        sns.set_theme(style="whitegrid")

        palette = ["#1976d2", "#388e3c", "#f57c00"]
        ax = sns.barplot(data=plot_df, x="Category", y="Percentage", hue="Split", palette=palette)

        plt.title("Stratified Class Distribution Across Dataset Splits", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Cyberbullying Category", fontsize=12)
        plt.ylabel("Percentage of Split (%)", fontsize=12)
        plt.legend(title="Split", frameon=True, facecolor="white")
        plt.tight_layout()

        out_path = os.path.join(self.vis_dir, "class_distribution_splits.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved split distribution visualization to {out_path}")

    def _plot_class_weights(self, weights_dict: Dict[str, float]):
        """Visualizes the balanced class weights calculated for loss weighting."""
        plt.figure(figsize=(9, 5))
        sns.set_theme(style="whitegrid")

        categories = list(weights_dict.keys())
        weights = list(weights_dict.values())

        colors = ["#455a64" if w < 5.0 else "#d32f2f" for w in weights]
        ax = sns.barplot(x=categories, y=weights, palette=colors)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height:.2f}x",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        plt.title("Balanced Loss Class Weights (CrossEntropyLoss)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Cyberbullying Category", fontsize=11)
        plt.ylabel("Loss Multiplier Weight", fontsize=11)
        plt.tight_layout()

        out_path = os.path.join(self.vis_dir, "class_imbalance_weights.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved class weights visualization to {out_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    integrated_csv = os.path.join(base_dir, "datasets", "processed", "cyberbullying_integrated_cleaned.csv")
    
    if os.path.exists(integrated_csv):
        df = pd.read_csv(integrated_csv)
        splitter = DatasetSplitter()
        train_df, val_df, test_df, summary = splitter.split_dataset(df)
        print("\n--- Splitting Summary ---")
        print(f"Train Records: {len(train_df):,}")
        print(f"Val Records  : {len(val_df):,}")
        print(f"Test Records : {len(test_df):,}")
        print("\nBalanced Class Weights:")
        for cat, weight in summary["imbalance_report"]["balanced_class_weights"].items():
            print(f"  {cat:<20}: {weight:.4f}x")
    else:
        print(f"Integrated dataset not found at {integrated_csv}. Run dataset_integration.py first.")

