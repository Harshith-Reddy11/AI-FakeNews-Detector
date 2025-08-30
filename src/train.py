# src/train.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys

# -------- Paths (robust, independent of where you run from) --------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PIPE_PATH = MODELS_DIR / "fake_news_pipeline.joblib"

# -------- Find dataset files (supports False.csv or Fake.csv) --------
true_path = DATA_DIR / "True.csv"

fake_candidates = [DATA_DIR / "False.csv", DATA_DIR / "Fake.csv"]
fake_path = next((p for p in fake_candidates if p.exists()), None)

if not true_path.exists():
    print(f"❌ Missing file: {true_path}")
    sys.exit(1)

if fake_path is None:
    print("❌ Missing file: data/False.csv (or data/Fake.csv)")
    sys.exit(1)

print(f"✅ Using TRUE file : {true_path.name}")
print(f"✅ Using FAKE file : {fake_path.name}")

# -------- Load CSVs --------
true_df = pd.read_csv(true_path)
fake_df = pd.read_csv(fake_path)

# -------- Pick text columns (title + text if available) --------
def pick_text(df):
    cols = {c.lower(): c for c in df.columns}  # case-insensitive map
    title_col = cols.get("title")
    text_col = cols.get("text") or cols.get("content") or cols.get("article") or cols.get("body")
    if text_col is None:
        raise ValueError(
            f"Could not find a text/content column. Available columns: {list(df.columns)}"
        )
    if title_col:
        return (df[title_col].astype(str) + " " + df[text_col].astype(str)).str.strip()
    return df[text_col].astype(str)

true_text = pick_text(true_df)
fake_text = pick_text(fake_df)

# -------- Label and combine --------
true_df = pd.DataFrame({"text": true_text, "label": 1})  # 1 = REAL
fake_df = pd.DataFrame({"text": fake_text, "label": 0})  # 0 = FAKE

df = pd.concat([true_df, fake_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"📦 Dataset size: {len(df)} (REAL={df.label.sum()}, FAKE={len(df)-df.label.sum()})")

# -------- Split (stratified) --------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# -------- Single Pipeline: TF-IDF + Logistic Regression --------
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),  # unigrams + bigrams for stronger baseline
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"  # helps if classes are imbalanced
    ))
])

print("🔧 Training...")
pipe.fit(X_train, y_train)

# -------- Evaluate --------
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

# -------- Save single artifact (pipeline includes vectorizer + model) --------
joblib.dump(pipe, PIPE_PATH)
print(f"\n💾 Saved pipeline to: {PIPE_PATH}")
