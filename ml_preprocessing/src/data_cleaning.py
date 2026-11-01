"""
Data Cleaning Module (BERT-Aware Preprocessing)
==============================================
Implements contextual, BERT-optimized NLP preprocessing.

Key Principles:
1. Preserve Context & Semantics: BERT is a contextual transformer model.
   Do NOT perform aggressive stopword removal, stemming, or lemmatization.
2. Preserve Negations & Slang: Negations ('not', 'never', 'no', 'can\'t')
   and slang expressions are essential for cyberbullying detection.
3. Emoji Demojization: Translate emojis into human-readable word tokens
   using `emoji.demojize()` so emotional cues (e.g. 😡 -> 'enraged face')
   are preserved and tokenized without triggering `[UNK]` tokens.
4. Entity Normalization: Replace URLs with '[URL]' and user handles with '[USER]'.
5. Hashtag Unpacking: Strip '#' to preserve the constituent vocabulary token.
6. Elongation Reduction: Cap repeated characters to 2 (e.g. 'sooooo' -> 'soo')
   to preserve emphasis while stabilizing WordPiece tokenization.
"""

import re
import html
import emoji
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple


def clean_text_bert(text: Any) -> str:
    """
    Cleans and normalizes a single text string while preserving BERT contextual features.
    
    Args:
        text: Input string or object.
        
    Returns:
        Cleaned string suitable for BERT tokenization.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities (&amp; -> &, &lt; -> <, &#039; -> ', &quot; -> ")
    cleaned = html.unescape(text)
    
    # 2. Strip HTML tags like <br>, <p>, </div>
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    
    # 3. Normalize newline and carriage return characters
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    
    # 4. Normalize URLs to a uniform token [URL]
    cleaned = re.sub(r"https?://\S+|www\.\S+", " [URL] ", cleaned)
    
    # 5. Normalize Twitter/social user mentions to [USER]
    cleaned = re.sub(r"@[\w_]+", " [USER] ", cleaned)
    
    # 6. Unpack hashtags: preserve semantic content (#cyberbullying -> cyberbullying)
    cleaned = re.sub(r"#([a-zA-Z0-9_]+)", r" \1 ", cleaned)
    
    # 7. Contextual Emoji preservation via Demojization
    # e.g., 😡 -> ' enraged face ', 🔪 -> ' kitchen knife '
    cleaned = emoji.demojize(cleaned, delimiters=(" ", " "))
    cleaned = cleaned.replace("_", " ")
    
    # 8. Character elongation normalization:
    # Reduce repetitions of 3+ consecutive identical characters to 2
    # e.g., "sooooo looooser" -> "soo looser" (retains intensity without OOV noise)
    cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
    
    # 9. Normalize excessive punctuation repetitions (e.g. '!!!!!!' -> '!!', '??????' -> '??')
    cleaned = re.sub(r"(!)\1+", "!!", cleaned)
    cleaned = re.sub(r"(\?)\1+", "??", cleaned)
    
    # 10. Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned


def is_valid_text(text: str, min_chars: int = 2) -> bool:
    """
    Validates whether cleaned text contains meaningful content.
    Filters out empty strings, pure whitespace, or strings without alphanumeric tokens.
    """
    if not text or len(text.strip()) < min_chars:
        return False
    # Check if there is at least one alphanumeric character
    if not re.search(r"[a-zA-Z0-9]", text):
        return False
    return True


class DataCleaner:
    """Batch data cleaning and deduplication pipeline for DataFrames."""

    def __init__(self, text_col: str = "text", min_chars: int = 2):
        self.text_col = text_col
        self.min_chars = min_chars

    def clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Cleans text column in a DataFrame, removes duplicates, and filters invalid rows.
        
        Returns:
            Cleaned DataFrame and audit dictionary tracking dropped records.
        """
        initial_count = len(df)
        df_out = df.copy()

        # Step 1: Remove rows with null text
        df_out = df_out.dropna(subset=[self.text_col])
        null_dropped = initial_count - len(df_out)

        # Step 2: Apply BERT-aware text cleaning
        df_out["cleaned_text"] = df_out[self.text_col].apply(clean_text_bert)

        # Step 3: Filter out empty or uninformative texts
        valid_mask = df_out["cleaned_text"].apply(lambda t: is_valid_text(t, min_chars=self.min_chars))
        df_out = df_out[valid_mask]
        invalid_dropped = (initial_count - null_dropped) - len(df_out)

        # Step 4: Deduplicate based on cleaned_text
        before_dedup = len(df_out)
        df_out = df_out.drop_duplicates(subset=["cleaned_text"])
        duplicates_dropped = before_dedup - len(df_out)

        audit_stats = {
            "initial_records": initial_count,
            "null_dropped": null_dropped,
            "invalid_dropped": invalid_dropped,
            "duplicates_dropped": duplicates_dropped,
            "final_clean_records": len(df_out),
        }

        return df_out, audit_stats


if __name__ == "__main__":
    sample_texts = [
        "RT @alex: You are sooooo dumb and ugly! 😡🖕 #loser https://t.co/xyz123",
        "I will literally kill you and burn your house down! 🔪💀",
        "Dear Jesus &amp; Mary, why is #MKR so white? &quot;crapilicious&quot;",
        "I love you so much! Have a blessed day &lt;3 ❤️",
        "     ",
        "???!!!",
        "RT @user: You are sooooo dumb and ugly! 😡🖕 #loser https://t.co/different_url",
    ]
    df_sample = pd.DataFrame({"text": sample_texts, "label_id": [1, 2, 0, 0, 0, 1, 1]})
    cleaner = DataCleaner(text_col="text")
    cleaned_df, stats = cleaner.clean_dataframe(df_sample)
    print("Audit Stats:\n", stats)
    print("\nCleaned Output:")
    for idx, row in cleaned_df.iterrows():
        print(f"[{row['label_id']}] {row['cleaned_text']}")
