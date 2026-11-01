"""
Label Mapping Module
====================
Provides formal taxonomies, mapping dictionaries, hierarchical resolvers,
and traceability records for harmonizing multi-source cyberbullying datasets
into the standard 5-class schema.

Target Classes:
    0: Not Cyberbullying
    1: Insult
    2: Threat
    3: Hate Speech
    4: Harassment
"""

from typing import Dict, Optional, Tuple, Any
import pandas as pd


# Canonical Target Taxonomy
LABEL_MAP = {
    0: "Not Cyberbullying",
    1: "Insult",
    2: "Threat",
    3: "Hate Speech",
    4: "Harassment",
}

LABEL_TO_ID = {name: idx for idx, name in LABEL_MAP.items()}

# Justification and definitions for documentation
LABEL_DEFINITIONS = {
    "Not Cyberbullying": (
        "Neutral, benign, informative, or supportive online text without abusive, "
        "derogatory, threatening, or harassing intent."
    ),
    "Insult": (
        "Explicitly offensive, belittling, or demeaning language targeted at an "
        "individual or entity's competence, appearance, or worth without invoking "
        "protected identity characteristics or bodily threats."
    ),
    "Threat": (
        "Statements conveying intent to inflict physical injury, violence, death, "
        "or severe harm to a person, their family, or their property."
    ),
    "Hate Speech": (
        "Language attacking, dehumanizing, or disparaging individuals or groups based on "
        "protected identity traits (race, ethnicity, religion, gender, sexual orientation, disability)."
    ),
    "Harassment": (
        "Targeted, hostile, repetitive, or covertly aggressive online attacks, "
        "trolling, cyber-stalking, or general interpersonal bullying."
    ),
}


class LabelMapper:
    """Encapsulates label mapping logic, validation, and traceability tracking."""

    @staticmethod
    def map_davidson(df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps Davidson et al. (2017):
            0 (Hate Speech)        -> Hate Speech (3)
            1 (Offensive Language) -> Insult (1)
            2 (Neither)            -> Not Cyberbullying (0)
        """
        df_out = df.copy()
        raw_to_target = {
            0: "Hate Speech",
            1: "Insult",
            2: "Not Cyberbullying",
        }
        df_out["original_dataset"] = "davidson_hate_speech"
        df_out["original_label"] = df_out["class"].astype(str)
        df_out["label_name"] = df_out["class"].map(raw_to_target)
        df_out["label_id"] = df_out["label_name"].map(LABEL_TO_ID)
        df_out["text"] = df_out["tweet"]
        return df_out[["text", "label_id", "label_name", "original_dataset", "original_label"]]

    @staticmethod
    def map_cyberbullying_tweets(df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps Wang et al. (2021) Cyberbullying Tweets:
            not_cyberbullying   -> Not Cyberbullying (0)
            religion            -> Hate Speech (3) [Targeted religious hate]
            ethnicity           -> Hate Speech (3) [Targeted racial/ethnic hate]
            gender              -> Hate Speech (3) [Targeted misogynistic/sexist hate]
            age                 -> Harassment (4)  [Targeted schoolyard/ageist harassment]
            other_cyberbullying -> Harassment (4)  [General interpersonal cyberbullying]
        """
        df_out = df.copy()
        raw_to_target = {
            "not_cyberbullying": "Not Cyberbullying",
            "religion": "Hate Speech",
            "ethnicity": "Hate Speech",
            "gender": "Hate Speech",
            "age": "Harassment",
            "other_cyberbullying": "Harassment",
        }
        df_out["original_dataset"] = "cyberbullying_tweets_kaggle"
        df_out["original_label"] = df_out["cyberbullying_type"]
        df_out["label_name"] = df_out["cyberbullying_type"].map(raw_to_target)
        df_out["label_id"] = df_out["label_name"].map(LABEL_TO_ID)
        df_out["text"] = df_out["tweet_text"]
        return df_out[["text", "label_id", "label_name", "original_dataset", "original_label"]]

    @staticmethod
    def map_jigsaw(df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps Jigsaw multi-label toxicity challenge using justified severity hierarchy:
            1. threat == 1         -> Threat (2) [Critical physical harm priority]
            2. identity_hate == 1  -> Hate Speech (3) [Protected group attacks]
            3. insult == 1         -> Insult (1) [Direct derogatory attacks]
            4. toxic / severe / obscene == 1 -> Harassment (4) [Pervasive general abuse]
            5. all zeros           -> Not Cyberbullying (0) [Clean talk comments]
        """
        df_out = df.copy()

        def resolve_jigsaw_row(row):
            if row["threat"] == 1:
                return "Threat", "threat=1"
            elif row["identity_hate"] == 1:
                return "Hate Speech", "identity_hate=1"
            elif row["insult"] == 1:
                return "Insult", "insult=1"
            elif row["toxic"] == 1 or row["severe_toxic"] == 1 or row["obscene"] == 1:
                return "Harassment", "toxic/severe/obscene=1"
            elif (
                row["toxic"] == 0
                and row["severe_toxic"] == 0
                and row["obscene"] == 0
                and row["threat"] == 0
                and row["insult"] == 0
                and row["identity_hate"] == 0
            ):
                return "Not Cyberbullying", "all_zero_clean"
            return None, "ambiguous"

        results = df_out.apply(resolve_jigsaw_row, axis=1)
        df_out["label_name"] = [r[0] for r in results]
        df_out["original_label"] = [r[1] for r in results]
        df_out["label_id"] = df_out["label_name"].map(LABEL_TO_ID)
        df_out["original_dataset"] = "jigsaw_toxic_comments"
        df_out["text"] = df_out["comment_text"]
        
        # Drop any ambiguous rows if present
        df_out = df_out.dropna(subset=["label_name", "label_id"])
        df_out["label_id"] = df_out["label_id"].astype(int)
        return df_out[["text", "label_id", "label_name", "original_dataset", "original_label"]]

    @staticmethod
    def map_trac1(df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps TRAC-1 Aggression Identification:
            NAG (Non-Aggressive)     -> Not Cyberbullying (0)
            CAG (Covertly Aggressive)-> Harassment (4) [Indirect passive-aggressive bullying]
            OAG (Overtly Aggressive) -> Insult (1)    [Direct offensive attack]
        """
        df_out = df.copy()
        raw_to_target = {
            "NAG": "Not Cyberbullying",
            "CAG": "Harassment",
            "OAG": "Insult",
        }
        df_out["original_dataset"] = "trac1_aggression"
        df_out["original_label"] = df_out["label"]
        df_out["label_name"] = df_out["label"].map(raw_to_target)
        df_out["label_id"] = df_out["label_name"].map(LABEL_TO_ID)
        df_out["text"] = df_out["text"]
        return df_out[["text", "label_id", "label_name", "original_dataset", "original_label"]]

    @staticmethod
    def inspect_formspring_limitation(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Documents the structural limitation of Formspring.me:
        Formspring provides crowd-annotated binary cyberbullying flags (Yes/No),
        but lacks multi-class categorization into Insult, Threat, Hate Speech,
        or Harassment. Forcing binary labels into granular classes violates the
        principle against forcing ambiguous labels.
        """
        total = len(df)
        yes_count = ((df["ans1"] == "Yes") | (df["ans2"] == "Yes") | (df["ans3"] == "Yes")).sum()
        no_count = ((df["ans1"] == "No") & (df["ans2"] == "No") & (df["ans3"] == "No")).sum()
        return {
            "dataset": "Formspring.me (Reynolds et al., 2011)",
            "total_records": total,
            "binary_bullying_positive": int(yes_count),
            "binary_bullying_negative": int(no_count),
            "limitation": (
                "Dataset schema only supports binary cyberbullying indicators without "
                "sub-categorization into Threat, Insult, Hate Speech, or Harassment. "
                "Documented as incompatible for 5-class multi-class integration."
            ),
            "action": "Excluded from 5-class combined dataset; retained in comparative analysis.",
        }


if __name__ == "__main__":
    from dataset_loader import DatasetLoader
    loader = DatasetLoader()
    raws = loader.load_all_raw()
    
    print("Testing Label Mapping...")
    d_dav = LabelMapper.map_davidson(raws["davidson"])
    print("Davidson mapped counts:\n", d_dav["label_name"].value_counts())
    
    d_cb = LabelMapper.map_cyberbullying_tweets(raws["cyberbullying_tweets"])
    print("\nCyberbullying Tweets mapped counts:\n", d_cb["label_name"].value_counts())
    
    d_jig = LabelMapper.map_jigsaw(raws["jigsaw_toxic"])
    print("\nJigsaw mapped counts:\n", d_jig["label_name"].value_counts())
    
    d_trac = LabelMapper.map_trac1(raws["trac1_aggression"])
    print("\nTRAC-1 mapped counts:\n", d_trac["label_name"].value_counts())
    
    fs_info = LabelMapper.inspect_formspring_limitation(raws["formspring"])
    print("\nFormspring Limitation Documented:\n", fs_info)

