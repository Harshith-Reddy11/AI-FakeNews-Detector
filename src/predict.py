# src/predict.py
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
PIPE_PATH = BASE_DIR / "models" / "fake_news_pipeline.joblib"

if not PIPE_PATH.exists():
    raise FileNotFoundError(f"Model not found at {PIPE_PATH}. Run: python src/train.py")

pipe = joblib.load(PIPE_PATH)

def predict_news(text: str) -> str:
    pred = pipe.predict([text])[0]
    return "✅ REAL News" if pred == 1 else "❌ FAKE News"

if __name__ == "__main__":
    print("📰 Fake News Detector (type 'exit' to quit)\n")
    while True:
        inp = input("Enter news text: ")
        if inp.strip().lower() == "exit":
            print("👋 Bye!")
            break
        print(predict_news(inp), "\n")
