from datasets import load_dataset
import pandas as pd
from pathlib import Path


OUTPUT_PATH = Path("data/processed/bitext_full.csv")


def main():
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

    df = dataset["train"].to_pandas()

    useful_columns = ["instruction", "response", "category"]

    missing_columns = [col for col in useful_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df = df[useful_columns].dropna()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved full dataset to {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nCategory counts:")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()