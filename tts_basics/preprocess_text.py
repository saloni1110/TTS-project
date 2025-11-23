import pandas as pd
import re
import ast

# Define your allowed characters set and mapping to integer IDs
characters = "abcdefghijklmnopqrstuvwxyz1234567890' "
char_to_id = {c: i + 1 for i, c in enumerate(characters)}  # 0 reserved for padding/unknown

def normalize_text(text):
    """
    Normalize text by lowercasing, stripping spaces,
    removing unwanted punctuation, and reducing spaces.
    """
    text = text.lower()
    text = text.strip()
    # Remove all characters except allowed (letters, digits, apostrophe, space)
    text = re.sub(r"[^a-z0-9' ]+", "", text)
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    return text

def text_to_sequence(text):
    """
    Convert normalized text to a list of character IDs.
    Unknown characters mapped to 0.
    """
    return [char_to_id.get(c, 0) for c in text]

def main():
    # Path to metadata
    metadata_path = "metadata.csv"

    # Load metadata CSV with pipe delimiter and no header
    metadata = pd.read_csv(metadata_path, sep="|", header=None, names=["filename", "transcript"])

    # Normalize transcripts
    metadata["normalized_transcript"] = metadata["transcript"].apply(normalize_text)

    # Convert normalized text to sequences
    metadata["sequence"] = metadata["normalized_transcript"].apply(text_to_sequence)

    # Save processed metadata including sequences to CSV
    metadata.to_csv("processed_metadata.csv", index=False)

    # Print a sample
    print(metadata.head())

if __name__ == "__main__":
    main()
