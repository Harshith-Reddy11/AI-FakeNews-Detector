from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model
model_path = "../models/transformer"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)


def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "REAL" if prediction == 1 else "FAKE"


# Example usage
if __name__ == "__main__":
    user_input = input("Enter news text: ")
    print(predict_news(user_input))
