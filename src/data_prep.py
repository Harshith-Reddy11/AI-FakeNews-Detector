import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def load_dataset(true_path="../data/True.csv", fake_path="../data/Fake.csv"):
    # Load datasets
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    # Add labels (1 = true, 0 = fake)
    true_df["label"] = 1
    fake_df["label"] = 0

    # Combine
    df = pd.concat([true_df, fake_df], ignore_index=True)

    # Use only text + label
    df = df[["text", "label"]]

    # Train-test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
    test_df = pd.DataFrame({"text": test_texts, "label": test_labels})

    # Convert to Hugging Face dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset)
