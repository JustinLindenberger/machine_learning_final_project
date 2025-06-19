import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import emoji
import contractions
from emoji import demojize
from wordsegment import load, segment

# this load function is needed by the contractions package.
load()

df = pd.read_csv("twitter_validation.csv", header=None)
df = df.iloc[:, [2, 3]]
df.columns = ["label", "tweets"]
df = df[df["label"] != "Irrelevant"]  # Drop irrelevant
label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df["label_id"] = df["label"].map(label_map)

df = df.dropna(subset=["tweets"])

def hashtag_to_words(tag):
    tag_body = tag.lstrip('#')
    words = segment(tag_body)
    return " ".join(words) if words else tag_body

# Text cleaning function
def clean_text(text):
    # Remove URLs, mentions, hashtags, and control characters
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'") \
            .replace("“", '"').replace("”", '"')
    text = re.sub(r"\.{3,}", " threeconsecutivedots ", text)
    text = re.sub(r"…", " threeconsecutivedots ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r'\b(?:[a-z][a-z0-9+\-.]*://|www\.)\S+\b', 'URLLINK', text)
    text = re.sub(r'@\w+', 'ATUSERNME', text)
    text = re.sub(r'\d{2,}', 'NUMBERSEQUENCE', text)
    # text = re.sub(hashtag_pattern, '', text)
    text = re.sub(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Hashtags → word sequences
    # do a regex “find all” and replace each with our function
    text = re.sub(r'#\w+', lambda m: hashtag_to_words(m.group(0)), text)

    # Possessive 's → plain s
    text = re.sub(r"(?<=\w)'s\b", "s", text)

    # text = contr_pat.sub(replace_contraction, text)
    text = contractions.fix(text)

    # Convert emojis to text descriptions (do not delete them)
    text = demojize(text, language='en')  # 🤯 → :exploding_head:

    # Replace :emoji_name: with emoji:emoji_name: using regex
    text = re.sub(r':([a-zA-Z0-9_+-]+):', r'emoji:\1:', text)

    # Remove newline and escaped newline characters (\n and \\n)
    text = re.sub(r'(\\n|\n)', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply the cleaning function
df["tweets"] = df["tweets"].astype(str).apply(clean_text)

# Compute tweet lengths in tokens (words)
lengths = df["tweets"].astype(str).str.split().apply(len)

# Summary statistics
mean_len   = lengths.mean()
median_len = lengths.median()
min_len    = lengths.min()
max_len    = lengths.max()
std_len    = lengths.std()

print(f"Mean tweet length:   {mean_len:.2f} tokens")
print(f"Median tweet length: {median_len} tokens")
print(f"Min tweet length:    {min_len} tokens")
print(f"Max tweet length:    {max_len} tokens")
print(f"Std dev of lengths:  {std_len:.2f} tokens")

# Histogram
plt.figure(figsize=(8,4))
plt.hist(lengths, bins=30)
plt.title("Distribution of Tweet Lengths (in Tokens)")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Save the preprocessed result
df.to_csv("file_preprocessed_transfer_dataset.csv", index=False, encoding='utf-8-sig')
print("Preprocessing complete. Saved as file_preprocessed_transfer_dataset.csv")