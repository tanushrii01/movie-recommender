"""
Dataset Loader Module
=====================
Handles downloading, caching, loading, and initial structural validation
of the 5 benchmark cyberbullying and online toxicity datasets.

Datasets Managed:
1. Davidson et al. (2017) - Hate Speech and Offensive Language (Twitter)
2. Wang et al. (2021) - Cyberbullying Classification Dataset (Twitter)
3. Jigsaw / Google Conversation AI (2018) - Toxic Comment Classification (Wikipedia Talk Pages)
4. Kumar et al. (2018) - TRAC-1 Aggression Identification Dataset (Facebook Comments)
5. Reynolds et al. (2011) - Formspring.me Cyberbullying Dataset (Q&A Posts)
"""

import os
import urllib.request
import time
import pandas as pd
from typing import Dict, Tuple, Optional


DATASET_METADATA = {
    "davidson": {
        "name": "Davidson et al. (2017) Hate Speech & Offensive Language",
        "filename": "davidson_hate_speech.csv",
        "url": "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
        "source": "Davidson et al., ICWSM 2017",
        "domain": "Twitter",
        "license": "MIT License",
        "text_column": "tweet",
        "raw_label_column": "class",
        "size_expected_approx": 2500000,
    },
    "cyberbullying_tweets": {
        "name": "Wang et al. (2021) Multi-Class Cyberbullying Tweets",
        "filename": "cyberbullying_tweets.csv",
        "url": "https://raw.githubusercontent.com/am-shb/cyberbullying-detection/main/data/cyberbullying_tweets.csv",
        "source": "Kaggle / IEEE (Wang et al., 2021)",
        "domain": "Twitter",
        "license": "CC0: Public Domain / Open Research",
        "text_column": "tweet_text",
        "raw_label_column": "cyberbullying_type",
        "size_expected_approx": 7100000,
    },
    "jigsaw_toxic": {
        "name": "Jigsaw Toxic Comment Classification Challenge",
        "filename": "jigsaw_toxic_comments.csv",
        "url": "https://huggingface.co/datasets/Arsive/toxicity_classification_jigsaw/resolve/main/train_dataset.csv",
        "source": "Jigsaw & Google Conversation AI / Kaggle (2018)",
        "domain": "Wikipedia Talk Pages",
        "license": "CC0: Public Domain",
        "text_column": "comment_text",
        "raw_label_column": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        "size_expected_approx": 10000000,
    },
    "trac1_aggression": {
        "name": "Kumar et al. (2018) TRAC-1 Aggression Identification",
        "filename": "trac1_aggression.csv",
        "url": "https://raw.githubusercontent.com/kmi-linguistics/trac-1/master/english/agr_en_train.csv",
        "source": "TRAC@COLING 2018 Shared Task (Kumar et al.)",
        "domain": "Facebook User Comments",
        "license": "CC-BY 4.0",
        "text_column": "text",
        "raw_label_column": "label",
        "size_expected_approx": 2100000,
    },
    "formspring": {
        "name": "Reynolds et al. (2011) Formspring.me Cyberbullying Dataset",
        "filename": "formspring_cyberbullying.csv",
        "url": "https://raw.githubusercontent.com/eimearfoley/CyberBullyingDetection/master/data/formspring.csv",
        "source": "Reynolds et al., ICMLA 2011 / Insight Centre",
        "domain": "Formspring.me Q&A Posts",
        "license": "Academic Research Use",
        "text_column": "post",
        "raw_label_column": ["ans1", "ans2", "ans3"],
        "size_expected_approx": 3900000,
    },
}


class DatasetLoader:
    """Manages downloading, caching, and loading of raw cyberbullying datasets."""

    def __init__(self, raw_data_dir: Optional[str] = None):
        if raw_data_dir is None:
            # Default to relative path ml_preprocessing/datasets/raw
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.raw_data_dir = os.path.join(base_dir, "datasets", "raw")
        else:
            self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def download_dataset(self, key: str, force_download: bool = False) -> str:
        """Downloads a single dataset if not already present."""
        if key not in DATASET_METADATA:
            raise ValueError(f"Unknown dataset key: {key}. Available: {list(DATASET_METADATA.keys())}")
        
        meta = DATASET_METADATA[key]
        dest_path = os.path.join(self.raw_data_dir, meta["filename"])
        
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0 and not force_download:
            return dest_path
        
        print(f"Downloading [{meta['name']}] from {meta['url']}...")
        req = urllib.request.Request(meta["url"], headers={"User-Agent": "Mozilla/5.0"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out_f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
        elapsed = time.time() - t0
        size_bytes = os.path.getsize(dest_path)
        print(f"Saved {meta['filename']} ({size_bytes:,} bytes in {elapsed:.2f}s)")
        return dest_path

    def download_all(self, force_download: bool = False) -> Dict[str, str]:
        """Downloads all 5 benchmark datasets."""
        paths = {}
        for key in DATASET_METADATA:
            paths[key] = self.download_dataset(key, force_download=force_download)
        return paths

    def load_davidson(self) -> pd.DataFrame:
        """Loads Davidson et al. (2017) hate speech dataset."""
        path = self.download_dataset("davidson")
        df = pd.read_csv(path)
        return df

    def load_cyberbullying_tweets(self) -> pd.DataFrame:
        """Loads Wang et al. (2021) Cyberbullying Tweets dataset."""
        path = self.download_dataset("cyberbullying_tweets")
        df = pd.read_csv(path)
        return df

    def load_jigsaw(self) -> pd.DataFrame:
        """Loads Jigsaw Toxic Comment Classification Challenge dataset."""
        path = self.download_dataset("jigsaw_toxic")
        df = pd.read_csv(path)
        return df

    def load_trac1(self) -> pd.DataFrame:
        """Loads TRAC-1 Aggression Identification dataset."""
        path = self.download_dataset("trac1_aggression")
        # TRAC-1 has no header row
        df = pd.read_csv(path, header=None, names=["id", "text", "label"])
        return df

    def load_formspring(self) -> pd.DataFrame:
        """Loads Formspring.me Cyberbullying dataset."""
        path = self.download_dataset("formspring")
        df = pd.read_csv(path)
        return df

    def load_all_raw(self) -> Dict[str, pd.DataFrame]:
        """Loads all 5 raw datasets into a dictionary of DataFrames."""
        return {
            "davidson": self.load_davidson(),
            "cyberbullying_tweets": self.load_cyberbullying_tweets(),
            "jigsaw_toxic": self.load_jigsaw(),
            "trac1_aggression": self.load_trac1(),
            "formspring": self.load_formspring(),
        }

    def inspect_dataset_summary(self) -> pd.DataFrame:
        """Generates a tabular comparative summary of all 5 datasets."""
        raws = self.load_all_raw()
        summary_rows = []
        for key, df in raws.items():
            meta = DATASET_METADATA[key]
            text_col = meta["text_column"]
            null_text_count = df[text_col].isnull().sum() if text_col in df.columns else 0
            dup_count = df.duplicated(subset=[text_col]).sum() if text_col in df.columns else df.duplicated().sum()
            summary_rows.append({
                "Dataset Key": key,
                "Dataset Name": meta["name"],
                "Source / Paper": meta["source"],
                "Platform": meta["domain"],
                "License": meta["license"],
                "Total Rows": len(df),
                "Total Columns": len(df.columns),
                "Text Column": text_col,
                "Missing Texts": null_text_count,
                "Duplicate Texts": dup_count,
                "Labels Description": str(meta["raw_label_column"]),
            })
        return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    loader = DatasetLoader()
    print("Verifying downloads and loading datasets...")
    summary_df = loader.inspect_dataset_summary()
    print(summary_df.to_string(index=False))

