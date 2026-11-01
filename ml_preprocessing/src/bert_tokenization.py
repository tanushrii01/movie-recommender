"""
BERT Tokenization Module
========================
Prepares the cleaned, integrated cyberbullying dataset for BERT fine-tuning
using `bert-base-uncased` and `BertTokenizer` from Hugging Face Transformers.

Capabilities:
1. Sequence Length Distribution Analysis: Calculates token length statistics
   (mean, median, 90th, 95th, 99th percentiles) and exports visualization.
2. Max Sequence Length Selection: Justified recommendations balancing context
   coverage vs quadratic transformer memory complexity.
3. Tokenization Demonstration: Step-by-step tokenization walkthrough across
   all 5 target categories (tokens, input_ids, attention_mask, decoding).
4. PyTorch Dataset Implementation: `CyberbullyingBERTDataset` ready for
   PyTorch DataLoader and Hugging Face Trainer in model training.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertTokenizer

# Model architecture specification
BERT_MODEL_NAME = "bert-base-uncased"


class CyberbullyingBERTDataset(Dataset):
    """
    PyTorch Dataset wrapper for BERT cyberbullying classification.
    Produces tokenized tensors ready for Hugging Face / PyTorch models.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[BertTokenizerFast] = None,
        max_length: int = 128,
    ):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer or BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class BertTokenizationPipeline:
    """Manages BERT tokenization analysis, demonstrations, and dataset generation."""

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        output_dir: Optional[str] = None,
        vis_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = output_dir or os.path.join(base_dir, "outputs")
        self.vis_dir = vis_dir or os.path.join(base_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        print(f"Loading {self.model_name} tokenizer...")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)

    def analyze_sequence_lengths(
        self,
        texts: List[str],
        sample_size: int = 25000,
        save_plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyzes the distribution of WordPiece token lengths across texts.
        Computes descriptive statistics and percentiles to inform max_length selection.
        """
        print(f"Analyzing token lengths for {min(len(texts), sample_size):,} sample texts...")
        if len(texts) > sample_size:
            np.random.seed(42)
            eval_texts = np.random.choice(texts, size=sample_size, replace=False)
        else:
            eval_texts = texts

        # Tokenize without padding or truncation to measure natural lengths
        token_lengths = [len(self.tokenizer.encode(str(t), truncation=False)) for t in eval_texts]
        lengths_arr = np.array(token_lengths)

        stats = {
            "total_analyzed": len(lengths_arr),
            "mean_length": float(np.mean(lengths_arr)),
            "std_length": float(np.std(lengths_arr)),
            "median_length (p50)": float(np.median(lengths_arr)),
            "p75_length": float(np.percentile(lengths_arr, 75)),
            "p90_length": float(np.percentile(lengths_arr, 90)),
            "p95_length": float(np.percentile(lengths_arr, 95)),
            "p98_length": float(np.percentile(lengths_arr, 98)),
            "p99_length": float(np.percentile(lengths_arr, 99)),
            "max_length": int(np.max(lengths_arr)),
            "coverage_at_64": float(np.mean(lengths_arr <= 64) * 100),
            "coverage_at_128": float(np.mean(lengths_arr <= 128) * 100),
            "coverage_at_160": float(np.mean(lengths_arr <= 160) * 100),
            "coverage_at_256": float(np.mean(lengths_arr <= 256) * 100),
            "recommended_max_length": 128 if np.mean(lengths_arr <= 128) >= 0.95 else 160,
        }

        # Save stats to outputs
        stats_path = os.path.join(self.output_dir, "token_length_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        if save_plot:
            self._plot_length_distribution(lengths_arr, stats)

        return stats

    def _plot_length_distribution(self, lengths: np.ndarray, stats: Dict[str, Any]):
        """Generates and saves token length distribution plot."""
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # Cap x-axis for clean visualization (most social texts are < 200 tokens)
        display_lengths = np.clip(lengths, 0, 200)

        sns.histplot(
            display_lengths,
            kde=True,
            bins=50,
            color="#2b5c8f",
            edgecolor="white",
            alpha=0.8,
        )

        p95 = stats["p95_length"]
        rec = stats["recommended_max_length"]

        plt.axvline(p95, color="#e65100", linestyle="--", linewidth=2, label=f"95th Percentile ({p95:.0f} tokens)")
        plt.axvline(rec, color="#2e7d32", linestyle="-", linewidth=2.5, label=f"Recommended max_length = {rec} ({stats[f'coverage_at_{rec}']:.1f}% coverage)")

        plt.title("WordPiece Token Length Distribution (BERT-base-uncased)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Number of Tokens (including [CLS] and [SEP])", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend(frameon=True, facecolor="white", loc="upper right")
        plt.tight_layout()

        plot_path = os.path.join(self.vis_dir, "token_length_distribution.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Token length distribution plot saved to {plot_path}")

    def demonstrate_tokenization(self, sample_records: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Demonstrates tokenization across canonical categories with detailed representations.
        """
        if sample_records is None:
            sample_records = [
                {"label_name": "Not Cyberbullying", "text": "Had an amazing weekend with family! Blessed and thankful <3"},
                {"label_name": "Insult", "text": "RT [USER]: You are the absolute dumbest and most pathetic clown on this site!!"},
                {"label_name": "Threat", "text": "I will find where you live and end you and your family! kitchen knife skull"},
                {"label_name": "Hate Speech", "text": "These dirty immigrants and muslims should be expelled from our country immediately"},
                {"label_name": "Harassment", "text": "Nobody likes you loser, stop replying to everyone. Go away and never come back!!"},
            ]

        demonstrations = []
        print("\n" + "=" * 80)
        print("BERT-BASE-UNCASED TOKENIZATION DEMONSTRATION")
        print("=" * 80)

        for item in sample_records:
            text = item["text"]
            label = item["label_name"]

            tokens = self.tokenizer.tokenize(text)
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=24,
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True,
            )

            input_ids = encoded["input_ids"][0].tolist()
            attention_mask = encoded["attention_mask"][0].tolist()
            decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

            demo_entry = {
                "category": label,
                "input_text": text,
                "token_count": len(tokens),
                "wordpiece_tokens": tokens[:12],
                "input_ids_first_12": input_ids[:12],
                "attention_mask_first_12": attention_mask[:12],
                "special_tokens": {
                    "[CLS] id": self.tokenizer.cls_token_id,
                    "[SEP] id": self.tokenizer.sep_token_id,
                    "[PAD] id": self.tokenizer.pad_token_id,
                    "[UNK] id": self.tokenizer.unk_token_id,
                },
                "decoded_sample": decoded_text,
            }
            demonstrations.append(demo_entry)

            print(f"\n[Category: {label.upper()}]")
            print(f"Input Text    : \"{text}\"")
            print(f"Tokens (WordPiece): {tokens[:10]} ...")
            print(f"Input IDs     : {input_ids[:10]} ...")
            print(f"Attention Mask: {attention_mask[:10]} ...")
            print(f"Decoded Check : \"{self.tokenizer.decode(input_ids[:12])}\"")

        # Save demo to outputs
        demo_path = os.path.join(self.output_dir, "tokenization_demonstration.json")
        with open(demo_path, "w", encoding="utf-8") as f:
            json.dump(demonstrations, f, indent=2)

        return demonstrations


if __name__ == "__main__":
    pipeline = BertTokenizationPipeline()
    
    # Test on integrated dataset if present
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    integrated_csv = os.path.join(base_dir, "datasets", "processed", "cyberbullying_integrated_cleaned.csv")
    
    if os.path.exists(integrated_csv):
        df = pd.read_csv(integrated_csv)
        stats = pipeline.analyze_sequence_lengths(df["cleaned_text"].tolist(), sample_size=20000)
        print("\nSequence Length Statistics:")
        for k, v in stats.items():
            print(f"  {k:<25}: {v}")
    
    pipeline.demonstrate_tokenization()

