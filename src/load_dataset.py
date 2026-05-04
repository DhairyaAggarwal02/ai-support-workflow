from datasets import load_dataset
import pandas as pd

def main():
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

    train_data = dataset['train']

    # Convert to pandas
    df = train_data.to_pandas()

    # Keep only useful columns
    df = df[['instruction', 'response', 'category']]

    # Save small sample (for faster dev)
    df_sample = df.sample(1000, random_state=42)

    df_sample.to_csv("data/processed/bitext_sample.csv", index=False)

    print("Saved sample dataset")

if __name__ == "__main__":
    main()