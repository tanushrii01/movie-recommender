"""
Script to build notebooks/dataset_analysis.ipynb and notebooks/preprocessing.ipynb
with rich markdown explanations, code blocks, output previews, and diagrams.
"""

import os
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notebooks_dir = os.path.join(base_dir, "notebooks")
os.makedirs(notebooks_dir, exist_ok=True)

def create_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    path = os.path.join(notebooks_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Created notebook: {path}")

# ==============================================================================
# NOTEBOOK 1: dataset_analysis.ipynb
# ==============================================================================
cells_analysis = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Intelligent Cyberbullying Detection Framework\n",
            "## Module: Exploratory Dataset Analysis (Person 1 - Dataset Collection & Preprocessing)\n",
            "\n",
            "### Overview\n",
            "This notebook provides an in-depth exploratory data analysis (EDA) of **5 benchmark cyberbullying and online toxicity datasets** collected from public academic and repository sources (Kaggle, Hugging Face, GitHub).\n",
            "\n",
            "### Research Objective\n",
            "Examine the structure, labeling schemes, class distributions, missing values, duplicates, and compatibility of each dataset to harmonize them into a canonical 5-class cyberbullying taxonomy:\n",
            "1. **Not Cyberbullying** (Class 0)\n",
            "2. **Insult** (Class 1)\n",
            "3. **Threat** (Class 2)\n",
            "4. **Hate Speech** (Class 3)\n",
            "5. **Harassment** (Class 4)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Add src directory to path\n",
            "sys.path.append(os.path.abspath('../src'))\n",
            "from dataset_loader import DatasetLoader, DATASET_METADATA\n",
            "from label_mapping import LabelMapper, LABEL_MAP, LABEL_DEFINITIONS\n",
            "\n",
            "sns.set_theme(style='whitegrid')\n",
            "print('Environment and modules loaded successfully!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Dataset Collection & Metadata\n",
            "We inspect the provenance, platform domain, license, and raw schema for the 5 benchmark datasets."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "loader = DatasetLoader()\n",
            "summary_df = loader.inspect_dataset_summary()\n",
            "summary_df[['Dataset Key', 'Dataset Name', 'Platform', 'License', 'Total Rows', 'Missing Texts', 'Duplicate Texts']]"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. In-Depth Analysis of Individual Datasets\n",
            "### 2.1 Davidson et al. (2017) Hate Speech & Offensive Language\n",
            "- **Source**: Davidson et al. (ICWSM 2017), Twitter corpus.\n",
            "- **Classes**: 0: Hate Speech, 1: Offensive Language, 2: Neither."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_dav = loader.load_davidson()\n",
            "print(f'Total records: {len(df_dav):,}')\n",
            "print('Raw class distribution:')\n",
            "print(df_dav['class'].value_counts())\n",
            "df_dav[['tweet', 'class']].head(4)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.2 Wang et al. (2021) Cyberbullying Tweets\n",
            "- **Source**: Kaggle / IEEE (Wang et al., 2021), Twitter corpus.\n",
            "- **Classes**: `not_cyberbullying`, `religion`, `age`, `gender`, `ethnicity`, `other_cyberbullying`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_cb = loader.load_cyberbullying_tweets()\n",
            "print(f'Total records: {len(df_cb):,}')\n",
            "print('Raw class distribution:')\n",
            "print(df_cb['cyberbullying_type'].value_counts())\n",
            "df_cb.head(4)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.3 Jigsaw Toxic Comment Classification Challenge\n",
            "- **Source**: Google Conversation AI / Jigsaw (2018), Wikipedia Talk Pages.\n",
            "- **Multi-label flags**: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_jig = loader.load_jigsaw()\n",
            "print(f'Total records: {len(df_jig):,}')\n",
            "cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']\n",
            "print('Toxicity flag occurrences:')\n",
            "print(df_jig[cols].sum())\n",
            "df_jig[['comment_text'] + cols].head(4)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.4 TRAC-1 Aggression Identification\n",
            "- **Source**: TRAC@COLING 2018 Shared Task (Kumar et al.), Facebook user comments.\n",
            "- **Classes**: `NAG` (Non-Aggressive), `CAG` (Covertly Aggressive), `OAG` (Overtly Aggressive)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_trac = loader.load_trac1()\n",
            "print(f'Total records: {len(df_trac):,}')\n",
            "print('Raw class distribution:')\n",
            "print(df_trac['label'].value_counts())\n",
            "df_trac.head(4)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.5 Formspring.me Cyberbullying Dataset & Limitation Documentation\n",
            "- **Source**: Reynolds et al. (ICMLA 2011), Formspring.me Q&A posts.\n",
            "- **Limitation Analysis**: Formspring provides binary crowd annotations (`bully1..3`, `ans1..3`), but lacks the semantic granularity to classify into Insult, Threat, Hate Speech, or Harassment. Forcing binary labels into the 5 classes would violate the project's strict requirement against fabricating or forcing ambiguous labels."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_fs = loader.load_formspring()\n",
            "limitation_doc = LabelMapper.inspect_formspring_limitation(df_fs)\n",
            "print('Formspring Limitation Documented:')\n",
            "for k, v in limitation_doc.items():\n",
            "    print(f'  {k}: {v}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Label Harmonization & Cross-Dataset Integration\n",
            "We inspect the harmonized mapping results across the 4 compatible datasets."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from dataset_integration import DatasetIntegrator\n",
            "integrator = DatasetIntegrator()\n",
            "df_combined, raw_breakdown = integrator.load_and_map_all()\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.countplot(data=df_combined, x='label_name', hue='original_dataset')\n",
            "plt.title('Category Representation Across Original Data Sources', fontsize=13, fontweight='bold')\n",
            "plt.xlabel('Target Category')\n",
            "plt.ylabel('Count')\n",
            "plt.xticks(rotation=15)\n",
            "plt.legend(title='Source Dataset')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

# ==============================================================================
# NOTEBOOK 2: preprocessing.ipynb
# ==============================================================================
cells_preprocessing = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Intelligent Cyberbullying Detection Framework\n",
            "## Module: Data Preprocessing & BERT Tokenization Preparation (Person 1)\n",
            "\n",
            "### Overview\n",
            "This notebook details the **BERT-aware NLP preprocessing**, **tokenization analysis with `bert-base-uncased`**, **stratified data splitting**, and **class imbalance handling**."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import json\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "sys.path.append(os.path.abspath('../src'))\n",
            "from data_cleaning import clean_text_bert, DataCleaner\n",
            "from bert_tokenization import BertTokenizationPipeline, CyberbullyingBERTDataset\n",
            "from dataset_splitting import DatasetSplitter\n",
            "\n",
            "sns.set_theme(style='whitegrid')\n",
            "print('Preprocessing and BERT pipeline ready!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. BERT-Aware Contextual Text Cleaning\n",
            "BERT relies heavily on bidirectional contextual attention. Therefore:\n",
            "- **NO** aggressive stopword removal (pronouns and negations like *not, no, never* are crucial for insults/threats).\n",
            "- **NO** stemming or lemmatization (destroys inflected English grammar).\n",
            "- **Emoji Demojization**: Converts emojis (😡, 🔪) to text descriptions (`enraged face`, `kitchen knife`) to avoid `[UNK]` tokens while preserving high-intensity emotion."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "test_samples = [\n",
            "    'RT @john_doe: You are sooooo dumb and ugly! 😡🖕 #loser https://t.co/abc1234',\n",
            "    'I will find where you live and kill you! 🔪💀',\n",
            "    'Dear Jesus &amp; Mary, why is #MKR so white? &quot;crapilicious&quot;',\n",
            "    'I love everyone! Have a blessed day &lt;3 ❤️'\n",
            "]\n",
            "\n",
            "for s in test_samples:\n",
            "    print(f'RAW  : {s}')\n",
            "    print(f'CLEAN: {clean_text_bert(s)}')\n",
            "    print('-' * 70)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Integrated Dataset Inspection\n",
            "Load the preprocessed, deduplicated master corpus generated by `dataset_integration.py`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_clean = pd.read_csv('../datasets/processed/cyberbullying_integrated_cleaned.csv')\n",
            "print(f'Cleaned integrated dataset size: {len(df_clean):,} records')\n",
            "print('\\nClass distribution:')\n",
            "print(df_clean['label_name'].value_counts())\n",
            "df_clean[['cleaned_text', 'label_id', 'label_name', 'original_dataset']].head(5)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. BERT Tokenization Analysis (`bert-base-uncased`)\n",
            "We analyze the WordPiece token length distribution across the corpus to select an optimal `max_length`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "tok_pipeline = BertTokenizationPipeline()\n",
            "stats = tok_pipeline.analyze_sequence_lengths(df_clean['cleaned_text'].tolist(), sample_size=15000)\n",
            "\n",
            "for k, v in stats.items():\n",
            "    print(f'{k:<25}: {v}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3.1 Tokenization Demonstration Across Categories\n",
            "Examine how `BertTokenizer` encodes input IDs, attention masks, and special tokens (`[CLS]`, `[SEP]`, `[PAD]`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "demos = tok_pipeline.demonstrate_tokenization()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Stratified Dataset Splitting & Class Imbalance Analysis\n",
            "Dataset partitioned into **70% Train**, **15% Validation**, and **15% Testing**.\n",
            "Calculates balanced class weights for `nn.CrossEntropyLoss(weight=class_weights)` to handle the underrepresented `Threat` class without pruning majority data."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_df = pd.read_csv('../datasets/processed/train.csv')\n",
            "val_df = pd.read_csv('../datasets/processed/val.csv')\n",
            "test_df = pd.read_csv('../datasets/processed/test.csv')\n",
            "\n",
            "print(f'Train set: {len(train_df):,} samples ({len(train_df)/len(df_clean)*100:.1f}%)')\n",
            "print(f'Val set  : {len(val_df):,} samples ({len(val_df)/len(df_clean)*100:.1f}%)')\n",
            "print(f'Test set : {len(test_df):,} samples ({len(test_df)/len(df_clean)*100:.1f}%)')\n",
            "\n",
            "with open('../outputs/class_weights.json') as f:\n",
            "    weights_data = json.load(f)\n",
            "\n",
            "print('\\nBalanced Class Weights calculated for Model Loss:')\n",
            "for cat, weight in weights_data['balanced_class_weights'].items():\n",
            "    print(f'  {cat:<20}: {weight:.4f}x')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. PyTorch Dataset & DataLoader Verification\n",
            "Verify that `CyberbullyingBERTDataset` produces valid torch tensor batches for BERT fine-tuning."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "from torch.utils.data import DataLoader\n",
            "\n",
            "ds_train = CyberbullyingBERTDataset(\n",
            "    texts=train_df['cleaned_text'].head(100),\n",
            "    labels=train_df['label_id'].head(100),\n",
            "    tokenizer=tok_pipeline.tokenizer,\n",
            "    max_length=128\n",
            ")\n",
            "\n",
            "loader = DataLoader(ds_train, batch_size=16, shuffle=True)\n",
            "sample_batch = next(iter(loader))\n",
            "\n",
            "print('PyTorch Batch Tensors successfully constructed:')\n",
            "print('  input_ids shape     :', sample_batch['input_ids'].shape)\n",
            "print('  attention_mask shape:', sample_batch['attention_mask'].shape)\n",
            "print('  token_type_ids shape:', sample_batch['token_type_ids'].shape)\n",
            "print('  labels shape        :', sample_batch['labels'].shape)\n",
            "print('\\nReady for BERT model training by Person 2!')"
        ]
    }
]

create_notebook(cells_analysis, "dataset_analysis.ipynb")
create_notebook(cells_preprocessing, "preprocessing.ipynb")

