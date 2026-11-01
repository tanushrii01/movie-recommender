"""
Intelligent Cyberbullying Detection Framework - Master Pipeline Runner
=======================================================================
Executes the end-to-end Person 1 preprocessing pipeline:
1. Environment & Dependency Validation
2. Benchmark Dataset Acquisition & Inspection (5 datasets)
3. Label Harmonization into Canonical 5-Class Taxonomy
4. BERT-Aware Contextual Data Cleaning & Master Deduplication
5. BERT Tokenization Analysis with bert-base-uncased
6. Stratified Dataset Splitting (70% Train, 15% Val, 15% Test)
7. Class Imbalance Analysis & Balanced Loss Weights Computation
8. Publication Visualization Generation
9. PyTorch DataLoader Verification for Person 2 Model Training
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np

# Set environment variables for BERT and Transformers
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dataset_loader import DatasetLoader
from label_mapping import LabelMapper, LABEL_MAP
from data_cleaning import clean_text_bert, DataCleaner
from dataset_integration import DatasetIntegrator
from bert_tokenization import BertTokenizationPipeline, CyberbullyingBERTDataset
from dataset_splitting import DatasetSplitter


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_step(step_num: int, title: str):
    print(f"\n[STEP {step_num}] {title}")
    print("-" * 60)


def main():
    start_time = time.time()
    print_header("INTELLIGENT CYBERBULLYING DETECTION FRAMEWORK - PERSON 1 PIPELINE")
    print(f"Execution started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory   : {current_dir}")
    print(f"Python Executable   : {sys.executable}")
    print(f"Python Version      : {sys.version.split()[0]}")

    # --------------------------------------------------------------------------
    # Step 1: Environment & Dependency Validation
    # --------------------------------------------------------------------------
    print_step(1, "Verifying Environment & Core Libraries")
    libraries = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-Learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("emoji", "Emoji"),
    ]
    all_ok = True
    for mod_name, disp_name in libraries:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {disp_name:<30} (v{ver})")
        except ImportError as e:
            print(f"  [MISSING] {disp_name:<30} - {e}")
            all_ok = False

    if not all_ok:
        print("\nERROR: Required libraries are missing. Please run: pip install -r requirements.txt")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Step 2: Download & Ingest Benchmark Datasets
    # --------------------------------------------------------------------------
    print_step(2, "Acquiring & Inspecting 5 Public Benchmark Datasets")
    loader = DatasetLoader(raw_data_dir=os.path.join(current_dir, "datasets", "raw"))
    summary_df = loader.inspect_dataset_summary()
    print(summary_df[["Dataset Key", "Platform", "Total Rows", "Missing Texts", "Duplicate Texts"]].to_string(index=False))

    # --------------------------------------------------------------------------
    # Step 3: Label Mapping & Limitation Audit
    # --------------------------------------------------------------------------
    print_step(3, "Harmonizing Labels into 5-Class Taxonomy")
    raws = loader.load_all_raw()
    
    # Audit Formspring limitation
    fs_audit = LabelMapper.inspect_formspring_limitation(raws["formspring"])
    print(f"  Formspring Audit: {fs_audit['total_records']:,} records examined.")
    print(f"  Limitation Note : {fs_audit['limitation']}")
    print(f"  Action Taken    : {fs_audit['action']}")

    # --------------------------------------------------------------------------
    # Step 4: Contextual Preprocessing & Master Integration
    # --------------------------------------------------------------------------
    print_step(4, "Executing BERT-Aware Contextual Cleaning & Deduplication")
    integrator = DatasetIntegrator(
        output_dir=os.path.join(current_dir, "datasets", "processed")
    )
    # Re-run integration to ensure all files are fresh and synchronized
    integrated_df, audit_report = integrator.clean_and_integrate()
    print(f"  Initial combined raw records : {audit_report['total_raw_records']:,}")
    print(f"  Dropped duplicate texts      : {audit_report['audit_statistics']['duplicates_dropped']:,}")
    print(f"  Final unique clean records   : {audit_report['final_records']:,}")
    print("\n  Class Distribution in Cleaned Corpus:")
    for cat, cnt in audit_report["class_distribution"].items():
        pct = (cnt / audit_report["final_records"]) * 100
        print(f"    - {cat:<22}: {cnt:>6,} ({pct:>5.1f}%)")

    # --------------------------------------------------------------------------
    # Step 5: BERT Tokenization Sequence Analysis
    # --------------------------------------------------------------------------
    print_step(5, "Analyzing BERT Tokenization (bert-base-uncased)")
    tok_pipeline = BertTokenizationPipeline(
        model_name="bert-base-uncased",
        output_dir=os.path.join(current_dir, "outputs"),
        vis_dir=os.path.join(current_dir, "visualizations"),
    )
    token_stats = tok_pipeline.analyze_sequence_lengths(
        texts=integrated_df["cleaned_text"].tolist(),
        sample_size=20000,
        save_plot=True,
    )
    print(f"  Median Token Length (p50)   : {token_stats['median_length (p50)']:.1f} tokens")
    print(f"  95th Percentile Length (p95): {token_stats['p95_length']:.1f} tokens")
    print(f"  Coverage at max_length=128  : {token_stats['coverage_at_128']:.1f}%")
    print(f"  Recommended max_length      : {token_stats['recommended_max_length']} tokens")

    # Run concrete tokenization demonstration
    print("\n  Demonstrating Tokenization Across Categories:")
    tok_pipeline.demonstrate_tokenization()

    # --------------------------------------------------------------------------
    # Step 6: Stratified Splitting & Class Imbalance Analysis
    # --------------------------------------------------------------------------
    print_step(6, "Partitioning Stratified Splits & Calculating Class Weights")
    splitter = DatasetSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        output_dir=os.path.join(current_dir, "datasets", "processed"),
        vis_dir=os.path.join(current_dir, "visualizations"),
    )
    train_df, val_df, test_df, split_summary = splitter.split_dataset(
        integrated_df, text_col="cleaned_text", label_col="label_id"
    )
    print(f"  Training Split (70%)        : {len(train_df):,} records")
    print(f"  Validation Split (15%)      : {len(val_df):,} records")
    print(f"  Testing Split (15%)         : {len(test_df):,} records")
    print(f"  Data Leakage Status         : Leakage-Free = {split_summary['leakage_audit']['leakage_free']}")

    print("\n  Balanced Loss Weights for CrossEntropyLoss:")
    for cat, wt in split_summary["imbalance_report"]["balanced_class_weights"].items():
        print(f"    - {cat:<22}: {wt:.4f}x")

    # --------------------------------------------------------------------------
    # Step 7: Generate Visualizations
    # --------------------------------------------------------------------------
    print_step(7, "Rendering Visualizations")
    import generate_plots
    print("  Visualizations saved to visualizations/")

    # --------------------------------------------------------------------------
    # Step 8: PyTorch Dataset & DataLoader Verification
    # --------------------------------------------------------------------------
    print_step(8, "Verifying PyTorch DataLoader Batch Construction for Person 2")
    from torch.utils.data import DataLoader
    test_ds = CyberbullyingBERTDataset(
        texts=train_df["cleaned_text"].head(32).tolist(),
        labels=train_df["label_id"].head(32).tolist(),
        tokenizer=tok_pipeline.tokenizer,
        max_length=128,
    )
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    sample_batch = next(iter(loader))
    print(f"  DataLoader Batch Generated:")
    print(f"    - input_ids shape     : {list(sample_batch['input_ids'].shape)} [batch_size, max_length]")
    print(f"    - attention_mask shape: {list(sample_batch['attention_mask'].shape)}")
    print(f"    - token_type_ids shape: {list(sample_batch['token_type_ids'].shape)}")
    print(f"    - labels shape        : {list(sample_batch['labels'].shape)}")
    print("  [SUCCESS] Dataset is 100% plug-and-play for BERT model training!")

    # --------------------------------------------------------------------------
    # Pipeline Completed
    # --------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print_header("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Total Execution Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print("\nFinal Deliverables Summary:")
    print(f"  1. Master Cleaned Dataset : {os.path.join(current_dir, 'datasets', 'processed', 'cyberbullying_integrated_cleaned.csv')}")
    print(f"  2. Training Split (70%)   : {os.path.join(current_dir, 'datasets', 'processed', 'train.csv')}")
    print(f"  3. Validation Split (15%) : {os.path.join(current_dir, 'datasets', 'processed', 'val.csv')}")
    print(f"  4. Testing Split (15%)    : {os.path.join(current_dir, 'datasets', 'processed', 'test.csv')}")
    print(f"  5. Class Weights File     : {os.path.join(current_dir, 'outputs', 'class_weights.json')}")
    print(f"  6. Visualizations Folder  : {os.path.join(current_dir, 'visualizations')}")
    print(f"  7. Notebooks Folder       : {os.path.join(current_dir, 'notebooks')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

