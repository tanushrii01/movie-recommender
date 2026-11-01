# Intelligent Cyberbullying Detection Framework
## Module: Dataset Collection, Analysis, Integration & Preprocessing (Person 1)

---

### Table of Contents
1. [Project Overview & Objective](#1-project-overview--objective)
2. [Folder Structure](#2-folder-structure)
3. [Benchmark Dataset Collection & Sourcing](#3-benchmark-dataset-collection--sourcing)
4. [Dataset Comparison & Exploratory Analysis](#4-dataset-comparison--exploratory-analysis)
5. [Canonical Label Taxonomy & Mapping Justification](#5-canonical-label-taxonomy--mapping-justification)
6. [BERT-Aware Contextual Preprocessing Pipeline](#6-bert-aware-contextual-preprocessing-pipeline)
7. [BERT Tokenization Preparation (`bert-base-uncased`)](#7-bert-tokenization-preparation-bert-base-uncased)
8. [Class Imbalance Analysis & Loss Weighting](#8-class-imbalance-analysis--loss-weighting)
9. [Stratified Dataset Splitting (70% Train / 15% Val / 15% Test)](#9-stratified-dataset-splitting-70-train--15-val--15-test)
10. [Pipeline Execution Guide](#10-pipeline-execution-guide)
11. [Handoff Instructions for Person 2 (BERT Model Fine-Tuning)](#11-handoff-instructions-for-person-2-bert-model-fine-tuning)

---

## 1. Project Overview & Objective

The **Intelligent Cyberbullying Detection Framework** is an end-to-end NLP system designed to automatically detect, categorize, and moderate abusive digital communication across online platforms.

This repository encompasses the complete **Person 1 responsibility**:
- Sourcing 5 real-world public benchmark datasets (Twitter, Wikipedia Talk Pages, Facebook, Formspring).
- Conducting in-depth exploratory data analysis and labeling scheme harmonization.
- Performing BERT-aware contextual preprocessing (preserving negations, slang, and emotion via emoji demojization).
- Pre-computing tokenization sequence length distributions for `bert-base-uncased`.
- Partitioning into stratified, leakage-free Train (70%), Validation (15%), and Test (15%) splits.
- Computing balanced loss weights to resolve class imbalance (particularly for the rare `Threat` class).

---

## 2. Folder Structure

The preprocessing module is structured cleanly inside `ml_preprocessing/`:

```text
ml_preprocessing/
├── datasets/
│   ├── raw/                                # Pristine benchmark datasets downloaded from verified sources
│   │   ├── davidson_hate_speech.csv        # Davidson et al. (2017) (Twitter, 24,783 rows)
│   │   ├── cyberbullying_tweets.csv        # Wang et al. (2021) (Twitter, 47,692 rows)
│   │   ├── jigsaw_toxic_comments.csv       # Jigsaw / Conversation AI (Wikipedia, 25,960 rows)
│   │   ├── trac1_aggression.csv            # Kumar et al. (2018) TRAC-1 (Facebook, 12,000 rows)
│   │   └── formspring_cyberbullying.csv    # Reynolds et al. (2011) (Formspring, 12,901 rows)
│   └── processed/                          # Production-ready datasets for BERT fine-tuning
│       ├── cyberbullying_integrated_cleaned.csv   # Unified master corpus (107,613 unique records)
│       ├── cyberbullying_integrated_cleaned.jsonl # Unified JSON Lines format
│       ├── train.csv                       # 70% Stratified training split (75,329 records)
│       ├── val.csv                         # 15% Stratified validation split (16,142 records)
│       └── test.csv                        # 15% Stratified testing split (16,142 records)
├── notebooks/
│   ├── dataset_analysis.ipynb              # Comprehensive Exploratory Data Analysis (EDA) notebook
│   └── preprocessing.ipynb                 # Preprocessing walkthrough & tokenization demonstration
├── src/
│   ├── dataset_loader.py                   # Automated download, caching, and raw data ingestion
│   ├── data_cleaning.py                    # BERT-aware text cleaning, emoji demojization, normalization
│   ├── label_mapping.py                    # Schema mapping, multi-label resolvers, limitation audits
│   ├── dataset_integration.py              # Master consolidation, global deduplication, and export
│   ├── bert_tokenization.py                # BertTokenizerFast, sequence length stats, PyTorch Dataset
│   ├── dataset_splitting.py                # 70/15/15 Stratified splitting, leakage audits, loss weights
│   └── generate_plots.py                   # Generation of publication-quality charts
├── outputs/
│   ├── dataset_summary.json                # High-level dataset metadata, record counts, split audits
│   ├── integration_audit.json              # Cleaning drop statistics, cross-tabulations, source breakdown
│   ├── class_weights.json                  # Balanced loss weights formatted for PyTorch CrossEntropyLoss
│   ├── token_length_stats.json             # Percentile distribution of WordPiece token counts
│   └── tokenization_demonstration.json     # Concrete tokenization examples across all 5 classes
├── visualizations/
│   ├── class_distribution_raw.png          # Class distribution across the unified master dataset
│   ├── class_distribution_splits.png       # Stratified class distribution comparison across Train/Val/Test
│   ├── class_imbalance_weights.png         # Bar plot of computed loss multiplier weights
│   ├── token_length_distribution.png       # WordPiece token length histogram with p95 and max_length
│   ├── dataset_contribution.png            # Pie chart of source dataset contributions
│   └── dataset_category_heatmap.png        # Cross-tabulation heatmap of dataset sources vs categories
├── requirements.txt                        # Pinned dependencies
└── README.md                               # Preprocessing documentation & execution guide
```

---

## 3. Benchmark Dataset Collection & Sourcing

To guarantee academic authenticity and eliminate synthetic or fabricated data, five public benchmark datasets were identified, downloaded, and verified:

| # | Dataset Name | Authors / Source | Platform | License | Raw Records | Source URL |
|---|---|---|---|---|---|---|
| **1** | Automated Hate Speech and Offensive Language | Davidson et al. (ICWSM 2017) | Twitter | MIT License | 24,783 | [GitHub Repository](https://github.com/t-davidson/hate-speech-and-offensive-language) |
| **2** | Multi-Class Cyberbullying Classification | Wang et al. (IEEE / Kaggle 2021) | Twitter | CC0 / Public Domain | 47,692 | [GitHub Raw Mirror](https://raw.githubusercontent.com/am-shb/cyberbullying-detection/main/data/cyberbullying_tweets.csv) |
| **3** | Toxic Comment Classification Challenge | Jigsaw & Google Conversation AI (2018) | Wikipedia Talk Pages | CC0 / Public Domain | 25,960 | [Hugging Face Mirror](https://huggingface.co/datasets/Arsive/toxicity_classification_jigsaw) |
| **4** | TRAC-1 Aggression Identification Shared Task | Kumar et al. (COLING 2018) | Facebook Comments | CC-BY 4.0 | 12,000 | [Official TRAC-1 Repo](https://github.com/kmi-linguistics/trac-1) |
| **5** | Formspring.me Cyberbullying Dataset | Reynolds et al. (ICMLA 2011) | Formspring.me Q&A | Academic Research | 12,901 | [Insight Centre Data](https://raw.githubusercontent.com/eimearfoley/CyberBullyingDetection/master/data/formspring.csv) |

**Total Raw Records Collected**: **123,336 records**.

---

## 4. Dataset Comparison & Exploratory Analysis

### Comparative Summary Table

| Dataset Key | Platform | Modality | Text Column | Raw Labels Description | Missing Texts | Raw Duplicates | Suitable for 5-Class Integration? |
|---|---|---|---|---|---|---|---|
| `davidson` | Twitter | Short posts / tweets | `tweet` | 3 classes: 0 (Hate Speech), 1 (Offensive Language), 2 (Neither) | 0 | 0 | **Yes** (Direct 1:1 mapping) |
| `cyberbullying_tweets` | Twitter | Tweets & threads | `tweet_text` | 6 classes: `not_cyberbullying`, `religion`, `age`, `gender`, `ethnicity`, `other_cyberbullying` | 0 | 1,675 | **Yes** (Contextual mapping) |
| `jigsaw_toxic` | Wikipedia | Discussion comments | `comment_text` | Multi-label binary flags: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` | 0 | 0 | **Yes** (Hierarchical severity resolver) |
| `trac1_aggression` | Facebook | Public comments | `text` | 3 classes: `NAG` (Non-Aggressive), `CAG` (Covertly Aggressive), `OAG` (Overtly Aggressive) | 0 | 355 | **Yes** (Direct aggression mapping) |
| `formspring` | Formspring | Question-and-Answer pairs | `post` | 3 crowd annotator answers: `ans1`, `ans2`, `ans3` (Yes/No binary bullying) | 49 | 127 | **No** (Binary limitation; see below) |

### Formspring Limitation Analysis & Exclusion Justification
In accordance with the instruction (*"Before combining datasets, verify that their labels and meanings are compatible. If a dataset does not support reliable mapping into our five categories, document the limitation rather than forcing it"*), the Formspring.me dataset was analyzed in detail:
- Formspring provides **binary annotations** indicating whether bullying occurred (`Yes` vs `No`).
- It does **not** provide granular categorizations to distinguish whether an instance represents an *Insult*, a *Threat*, *Hate Speech*, or *Harassment*.
- Arbitrarily forcing binary positive cases into one of the four abusive sub-classes would inject substantial label noise and subjective bias into the training corpus.
- **Decision**: Formspring is retained in `datasets/raw/` and fully documented in `dataset_analysis.ipynb`, but **excluded from the unified 5-class integrated dataset**.

---

## 5. Canonical Label Taxonomy & Mapping Justification

The project targets a five-class taxonomy designed to cover the entire spectrum of online abuse:

| Label ID | Category Name | Definition & Contextual Scope |
|---|---|---|
| **0** | **Not Cyberbullying** | Benign, neutral, supportive, factual, or friendly social discourse free from abuse or hostility. |
| **1** | **Insult** | Derogatory, belittling, or demeaning language targeting an individual's appearance, intelligence, or character without invoking protected demographic classes or bodily threats. |
| **2** | **Threat** | Declarations conveying explicit or implicit intent to inflict physical violence, injury, murder, or property destruction on a person or their family. |
| **3** | **Hate Speech** | Hostile or dehumanizing attacks targeted at individuals or groups based on protected demographic identities (race, ethnicity, religion, gender, sexual orientation, disability). |
| **4** | **Harassment** | Repetitive, covert, ageist, predatory, or hostile cyber-mobbing, trolling, or targeted interpersonal bullying. |

### Mapping Rules by Source Dataset

#### 1. Davidson et al. (2017)
- `0 (Hate Speech)` $\rightarrow$ **Hate Speech (3)**: Direct slurs and dehumanizing language targeting demographic groups.
- `1 (Offensive Language)` $\rightarrow$ **Insult (1)**: Profanity, derogatory slurs, and demeaning remarks targeted at people.
- `2 (Neither)` $\rightarrow$ **Not Cyberbullying (0)**: Benign conversational text.

#### 2. Wang et al. (2021) Cyberbullying Tweets
- `not_cyberbullying` $\rightarrow$ **Not Cyberbullying (0)**
- `religion` $\rightarrow$ **Hate Speech (3)**: Slurs and hostility directed against religious groups (Islam, Judaism, Christianity).
- `ethnicity` $\rightarrow$ **Hate Speech (3)**: Racist slurs and derogatory tropes against racial/ethnic groups.
- `gender` $\rightarrow$ **Hate Speech (3)**: Misogynistic attacks, sexist slurs, and homophobic targeting.
- `age` $\rightarrow$ **Harassment (4)**: Targeted schoolyard and age-based bullying, mocking physical vulnerability.
- `other_cyberbullying` $\rightarrow$ **Harassment (4)**: General interpersonal harassment and repetitive online hostility.

#### 3. Jigsaw Toxic Comment Classification Challenge
Jigsaw uses multi-label binary indicators. We resolve overlaps using a **hierarchical severity priority rule**:
1. If `threat == 1` $\rightarrow$ **Threat (2)** (Physical harm threats take highest priority due to extreme safety risk).
2. Else if `identity_hate == 1` $\rightarrow$ **Hate Speech (3)** (Targeted identity attacks).
3. Else if `insult == 1` $\rightarrow$ **Insult (1)** (Explicit verbal degradation).
4. Else if `toxic == 1` or `severe_toxic == 1` or `obscene == 1` $\rightarrow$ **Harassment (4)** (Hostile abusive content not captured by threat/hate/insult).
5. Else if all flags are 0 $\rightarrow$ **Not Cyberbullying (0)** (Benign, unflagged Wikipedia talk comments).

#### 4. TRAC-1 Aggression Identification
- `NAG` (Non-Aggressive) $\rightarrow$ **Not Cyberbullying (0)**
- `CAG` (Covertly Aggressive) $\rightarrow$ **Harassment (4)** (Indirect bullying, subtle trolling, passive-aggressive hostility).
- `OAG` (Overtly Aggressive) $\rightarrow$ **Insult (1)** (Direct verbal aggression and abusive insults).

---

## 6. BERT-Aware Contextual Preprocessing Pipeline

Traditional NLP cleaning pipelines (e.g., removing all stopwords, aggressive Porter stemming, stripping all punctuation) severely harm BERT because transformer attention depends on syntax, pronouns, and sentence flow.

Our preprocessing pipeline (`data_cleaning.py`) is **custom-engineered for BERT**:

1. **HTML Entity Decoding**: Unescapes `&amp;` $\rightarrow$ `&`, `&lt;` $\rightarrow$ `<`, `&gt;` $\rightarrow$ `>`, `&#039;` $\rightarrow$ `'`, `&quot;` $\rightarrow$ `"` via `html.unescape`.
2. **HTML Tag Removal**: Strips stray markup (`<br>`, `<p>`, `</div>`) to clean scraped web artifacts.
3. **URL Normalization**: Replaces URLs with the uniform placeholder `[URL]` to eliminate web addresses without breaking sentence flow.
4. **User Mention Normalization**: Replaces `@username` with `[USER]` to protect privacy and prevent out-of-vocabulary handles from fragmenting into useless subword tokens.
5. **Hashtag Decomposition**: Strips `#` while retaining the constituent word (`#idiot` $\rightarrow$ `idiot`, `#cyberbullying` $\rightarrow$ `cyberbullying`) so BERT's WordPiece vocabulary can tokenize the semantic content.
6. **Contextual Emoji Demojization**: Uses `emoji.demojize()` to translate pictorial emojis into descriptive natural language tokens:
   - 😡 $\rightarrow$ `:enraged_face:` $\rightarrow$ `enraged face`
   - 🔪 $\rightarrow$ `:kitchen_knife:` $\rightarrow$ `kitchen knife`
   - 🖕 $\rightarrow$ `:middle_finger:` $\rightarrow$ `middle finger`
   - ❤️ $\rightarrow$ `:red_heart:` $\rightarrow$ `red heart`
   *Why this matters*: BERT's base vocabulary contains very few emoji codepoints and maps emojis to `[UNK]`. Demojizing translates raw emojis into words (`enraged`, `knife`, `middle`, `finger`) that BERT readily tokenizes, preserving critical emotional and bullying cues!
7. **Elongation Reduction**: Normalizes character repetitions of $\ge 3$ occurrences down to 2 (e.g., `sooooo dumb` $\rightarrow$ `soo dumb`, `looooser` $\rightarrow$ `looser`). This preserves the emotional emphasis without producing out-of-vocabulary anomalies.
8. **Negation & Slang Preservation**: Retains all negation tokens (`not`, `never`, `no`, `can't`, `won't`) and internet slang (`lmao`, `wtf`, `idc`).
9. **Global Deduplication & Cleaning Audit**: Drops exact duplicate cleaned strings across and within datasets to eliminate retweets and cross-corpus contamination.

### Data Cleaning Audit Statistics (from `integration_audit.json`)
- **Initial Combined Records**: 110,435
- **Null Records Dropped**: 0
- **Invalid / Empty Records Dropped**: 10
- **Duplicate Records Dropped**: 2,812
- **Final Clean Unique Records**: **107,613 records**

---

## 7. BERT Tokenization Preparation (`bert-base-uncased`)

The dataset is prepared specifically for `bert-base-uncased` with `BertTokenizerFast`.

### Sequence Length Distribution Analysis
Tokenizing across the unified corpus yielded the following WordPiece token statistics:

| Metric | WordPiece Tokens (including [CLS] and [SEP]) |
|---|---|
| **Mean Length** | 44.3 tokens |
| **Median Length (p50)** | 30.0 tokens |
| **75th Percentile (p75)** | 45.0 tokens |
| **90th Percentile (p90)** | 70.0 tokens |
| **95th Percentile (p95)** | 106.0 tokens |
| **98th Percentile (p98)** | 204.0 tokens |
| **99th Percentile (p99)** | 328.0 tokens |
| **Coverage at 64 tokens** | 87.10% |
| **Coverage at 128 tokens** | **96.09%** |
| **Coverage at 160 tokens** | **97.18%** |

### Maximum Sequence Length Recommendation: `max_length = 128`
- **96.1%** of all social media posts and comments fit entirely within **128 tokens**.
- Because BERT's self-attention complexity scales quadratically ($O(N^2)$), setting `max_length = 128` instead of 512 saves **75% of GPU VRAM** and increases training throughput by $\approx 3.5\times$ with virtually zero information loss.
- For high-memory training setups (e.g. A100 or V100), `max_length = 160` (97.2% coverage) can optionally be used.

### Concrete Tokenization Demonstration

```text
[Category: THREAT]
Input Text        : "I will find where you live and end you and your family! kitchen knife skull"
WordPiece Tokens  : ['i', 'will', 'find', 'where', 'you', 'live', 'and', 'end', 'you', 'and', 'your', 'family', '!', 'kitchen', 'knife', 'skull']
Input IDs         : [101, 1045, 2097, 2424, 2073, 2017, 2444, 1998, 2203, 2017, 1998, 2115, 2155, 999, 3829, 5442, 8040, 102, 0, 0, 0, 0, 0, 0]
Attention Mask    : [1,   1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,   1,    1,    1,    1,   0, 0, 0, 0, 0, 0]
Decoded Check     : "[CLS] i will find where you live and end you and your family! kitchen knife skull [SEP] [PAD] ..."
```

---

## 8. Class Imbalance Analysis & Loss Weighting

### Class Distribution Across the Master Dataset (N = 107,613)

| Label ID | Category Name | Total Samples | Percentage |
|:---:|---|:---:|:---:|
| **0** | Not Cyberbullying | 29,744 | 27.6% |
| **1** | Insult | 26,906 | 25.0% |
| **2** | Threat | 371 | 0.3% |
| **3** | Hate Speech | 26,105 | 24.3% |
| **4** | Harassment | 24,487 | 22.8% |

### Strategy for Handling Imbalance (The Rare `Threat` Category)
In real-world online discourse, violent threats are naturally low-frequency ($\approx 0.3\%$).
- **Blindly deleting majority classes was avoided**: Doing so would discard $>90,000$ high-quality training samples.
- **Solution**: We employ **Balanced Class Weighting** calculated via Scikit-learn:
  $$\text{weight}_c = \frac{N_{\text{train}}}{N_{\text{classes}} \times N_c}$$

The calculated loss weights (saved to `outputs/class_weights.json`) are:
- `Not Cyberbullying`: **0.7236x**
- `Insult`: **0.7999x**
- `Threat`: **57.9454x** *(Provides strong gradient signal on rare violent threats)*
- `Hate Speech`: **0.8245x**
- `Harassment`: **0.8789x**

---

## 9. Stratified Dataset Splitting (70% Train / 15% Val / 15% Test)

The master corpus was partitioned using **2-stage stratified sampling** (`random_state=42`):
- **Training Set (70%)**: **75,329 records** (`datasets/processed/train.csv`)
- **Validation Set (15%)**: **16,142 records** (`datasets/processed/val.csv`)
- **Testing Set (15%)**: **16,142 records** (`datasets/processed/test.csv`)

### Proportional Preservation Across Splits

| Category | Master Dataset (%) | Train Split (%) | Val Split (%) | Test Split (%) |
|---|:---:|:---:|:---:|:---:|
| Not Cyberbullying | 27.64% | 27.64% | 27.64% | 27.64% |
| Insult | 25.00% | 25.00% | 25.00% | 25.00% |
| Hate Speech | 24.26% | 24.26% | 24.26% | 24.26% |
| Harassment | 22.76% | 22.75% | 22.75% | 22.76% |
| Threat | 0.34% | 0.35% | 0.35% | 0.35% |

### Data Leakage Verification
- Exact text intersection test between Train, Val, and Test confirmed: **Zero overlapping texts (Overlap Count = 0)**.
- Data leakage status: **LEAKAGE-FREE**.

---

## 10. Pipeline Execution Guide

### Prerequisites
Install the required dependencies using the provided `requirements.txt`:
```bash
pip install -r ml_preprocessing/requirements.txt
```

### Reproducing the Pipeline from Scratch

1. **Step 1: Download & Ingest Raw Datasets**
   ```bash
   python ml_preprocessing/src/dataset_loader.py
   ```
   *Downloads all 5 datasets into `datasets/raw/` and prints a summary.*

2. **Step 2: Validate Label Mapping**
   ```bash
   python ml_preprocessing/src/label_mapping.py
   ```
   *Verifies taxonomy mapping and outputs Formspring limitation report.*

3. **Step 3: Run Text Cleaning & Master Integration**
   ```bash
   python ml_preprocessing/src/dataset_integration.py
   ```
   *Cleans, demojizes emojis, deduplicates, and creates `cyberbullying_integrated_cleaned.csv`.*

4. **Step 4: Analyze BERT Tokenization**
   ```bash
   python ml_preprocessing/src/bert_tokenization.py
   ```
   *Computes WordPiece token length distributions and generates token length plots.*

5. **Step 5: Perform Stratified Splitting & Class Weight Calculation**
   ```bash
   python ml_preprocessing/src/dataset_splitting.py
   ```
   *Splits data into `train.csv`, `val.csv`, `test.csv` and computes `class_weights.json`.*

6. **Step 6: Generate Publication Plots**
   ```bash
   python ml_preprocessing/src/generate_plots.py
   ```
   *Exports high-resolution charts to `visualizations/`.*

---

## 11. Handoff Instructions for Person 2 (BERT Model Fine-Tuning)

Person 2 can directly load the processed dataset and class weights into PyTorch or Hugging Face Trainer:

### Quickstart Code for Person 2:
```python
import json
import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader
from ml_preprocessing.src.bert_tokenization import CyberbullyingBERTDataset

# 1. Load Preprocessed Splits
train_df = pd.read_csv("ml_preprocessing/datasets/processed/train.csv")
val_df = pd.read_csv("ml_preprocessing/datasets/processed/val.csv")
test_df = pd.read_csv("ml_preprocessing/datasets/processed/test.csv")

# 2. Initialize Pretrained BertTokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
MAX_LENGTH = 128  # Covers 96.1% of sentences

# 3. Create PyTorch Datasets & DataLoaders
train_dataset = CyberbullyingBERTDataset(
    texts=train_df["cleaned_text"],
    labels=train_df["label_id"],
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 4. Load Balanced Class Weights for CrossEntropyLoss
with open("ml_preprocessing/outputs/class_weights.json") as f:
    weights_info = json.load(f)
weights_tensor = torch.tensor(weights_info["pytorch_tensor_weights_ordered"], dtype=torch.float)

# 5. Initialize Model & Weighted Criterion
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    id2label={
        0: "Not Cyberbullying",
        1: "Insult",
        2: "Threat",
        3: "Hate Speech",
        4: "Harassment"
    }
)
```

---
*Created by Person 1 for the Intelligent Cyberbullying Detection Framework project.*

