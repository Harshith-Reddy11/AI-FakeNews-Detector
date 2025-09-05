from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 🔹 Load your dataset (assuming data_prep.py saved a train/test split)
# Replace with the actual path/format of your dataset
dataset = load_dataset("csv", data_files={"train": "train_dataset", "test": "test_dataset"})

# 🔹 Take a small subset for FAST DEBUG
small_train = dataset["train"].shuffle(seed=42).select(range(100))
small_test = dataset["test"].shuffle(seed=42).select(range(50))

# 🔹 Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_data = small_train.map(tokenize, batched=True)
test_data = small_test.map(tokenize, batched=True)

# 🔹 Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# 🔹 Training args (FAST DEBUG MODE)
training_args = TrainingArguments(
    output_dir="./debug_results",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=10,
    save_steps=10,
    logging_steps=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    max_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 🔹 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 🔹 Run training (quick debug mode)
trainer.train()
