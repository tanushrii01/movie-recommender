"""
Dataset Integration Module
==========================
Harmonizes and integrates compatible benchmark cyberbullying datasets into
a unified, cleaned, and deduplicated master corpus.

Pipeline:
1. Loads compatible raw datasets: Davidson, Cyberbullying Tweets, Jigsaw, TRAC-1.
2. Standardizes schema: [text, label_id, label_name, original_dataset, original_label].
3. Concatenates into single master DataFrame.
4. Applies BERT-aware contextual cleaning (data_cleaning.py).
5. Performs global deduplication across datasets to prevent cross-dataset leakage.
6. Exports unified dataset to CSV and JSONL with metadata and provenance audit.
"""

import os
import json
import time
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataset_loader import DatasetLoader
from label_mapping import LabelMapper, LABEL_MAP, LABEL_TO_ID
from data_cleaning import DataCleaner


class DatasetIntegrator:
    """Manages multi-source dataset harmonization and integration."""

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.output_dir = os.path.join(base_dir, "datasets", "processed")
            self.report_dir = os.path.join(base_dir, "outputs")
        else:
            self.output_dir = output_dir
            self.report_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        self.loader = DatasetLoader()
        self.cleaner = DataCleaner(text_col="text", min_chars=2)

    def load_and_map_all(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads and maps the 4 compatible datasets into unified schema."""
        t0 = time.time()
        print("Loading raw datasets...")
        raw_dav = self.loader.load_davidson()
        raw_cb = self.loader.load_cyberbullying_tweets()
        raw_jig = self.loader.load_jigsaw()
        raw_trac = self.loader.load_trac1()

        print("Mapping labels to 5 canonical categories...")
        mapped_dav = LabelMapper.map_davidson(raw_dav)
        mapped_cb = LabelMapper.map_cyberbullying_tweets(raw_cb)
        mapped_jig = LabelMapper.map_jigsaw(raw_jig)
        mapped_trac = LabelMapper.map_trac1(raw_trac)

        raw_breakdown = {
            "davidson_hate_speech": len(mapped_dav),
            "cyberbullying_tweets_kaggle": len(mapped_cb),
            "jigsaw_toxic_comments": len(mapped_jig),
            "trac1_aggression": len(mapped_trac),
        }

        # Combine all mapped datasets
        df_combined = pd.concat([mapped_dav, mapped_cb, mapped_jig, mapped_trac], ignore_index=True)
        print(f"Combined {len(df_combined):,} raw mapped records from 4 datasets in {time.time() - t0:.2f}s")
        return df_combined, raw_breakdown

    def clean_and_integrate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Executes full integration: mapping, cleaning, and deduplication."""
        df_combined, raw_breakdown = self.load_and_map_all()

        print("Applying BERT-aware text cleaning & deduplication across corpus...")
        t0 = time.time()
        cleaned_df, audit_stats = self.cleaner.clean_dataframe(df_combined)
        print(f"Cleaning completed in {time.time() - t0:.2f}s")

        # Class distribution in final integrated dataset
        class_dist = cleaned_df["label_name"].value_counts().to_dict()
        dataset_dist = cleaned_df["original_dataset"].value_counts().to_dict()

        # Cross-tabulation of dataset vs category
        cross_tab = pd.crosstab(cleaned_df["original_dataset"], cleaned_df["label_name"]).to_dict()

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "raw_contributions": raw_breakdown,
            "total_raw_records": sum(raw_breakdown.values()),
            "audit_statistics": audit_stats,
            "final_records": len(cleaned_df),
            "class_distribution": class_dist,
            "dataset_distribution": dataset_dist,
            "cross_tabulation": cross_tab,
        }

        # Save integrated dataset files
        csv_path = os.path.join(self.output_dir, "cyberbullying_integrated_cleaned.csv")
        jsonl_path = os.path.join(self.output_dir, "cyberbullying_integrated_cleaned.jsonl")
        audit_path = os.path.join(self.report_dir, "integration_audit.json")

        print(f"Saving integrated dataset to {csv_path}...")
        # Columns to save: id, cleaned_text, label_id, label_name, original_dataset, original_label
        save_df = cleaned_df.reset_index(drop=True)
        save_df["id"] = save_df.index
        cols_order = ["id", "cleaned_text", "label_id", "label_name", "original_dataset", "original_label"]
        save_df = save_df[cols_order]

        save_df.to_csv(csv_path, index=False, encoding="utf-8")
        save_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"Integration complete! Saved {len(save_df):,} records.")
        print(f"Audit report written to {audit_path}")
        return save_df, report


if __name__ == "__main__":
    from typing import Optional
    integrator = DatasetIntegrator()
    final_df, audit = integrator.clean_and_integrate()
    print("\n--- Final Integration Summary ---")
    print(f"Total Unique Records: {audit['final_records']:,}")
    print("\nClass Distribution:")
    for label, count in audit["class_distribution"].items():
        pct = (count / audit["final_records"]) * 100
        print(f"  {label:<20}: {count:>6,} ({pct:>5.1f}%)")
    print("\nSource Dataset Contribution:")
    for ds, count in audit["dataset_distribution"].items():
        pct = (count / audit["final_records"]) * 100
        print(f"  {ds:<30}: {count:>6,} ({pct:>5.1f}%)")
