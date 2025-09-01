# 📰 AI Fake News Detector  

A Machine Learning project that classifies news articles as **Fake** or **True**.  
This project demonstrates text preprocessing, vectorization, and ML classification on real-world datasets.  

---

## 📂 Project Structure  

AI-FakeNews-Detector/
│
├── data/
│   ├── True.csv
│   ├── Fake.csv
│
├── notebooks/
│   ├── training.ipynb
│
├── src/
│   ├── train.py                 # Traditional TF-IDF training script
│   ├── predict.py               # Traditional TF-IDF prediction script
│   ├── data_prep.py             # Data preprocessing for transformer models
│   ├── train_transformer.py     # Initial transformer training pipeline (experimental)
│   ├── predict_transformer.py   # Transformer prediction script (experimental)
│
├── requirements.txt
├── README.md
├── .gitignore

---

## ⚙️ Tech Stack  

- **Language:** Python 🐍  
- **Libraries:**  
  - `pandas`, `numpy` – Data handling  
  - `scikit-learn` – ML algorithms (Logistic Regression)  
  - `nltk` – Text preprocessing  
  - `matplotlib`, `seaborn` – Visualization  

---

## 🚀 Features  

- Classifies news as **Fake** 🟥 or **True** 🟩  
- Preprocessing: stopword removal, stemming, lowercasing  
- TF-IDF Vectorization for feature extraction  
- Logistic Regression as baseline classifier  
- Works on custom user input  

---

## 🛠️ Installation  

Clone the repository:  
```bash
git clone https://github.com/Harshith-Reddy11/AI-FakeNews-Detector.git
cd AI-FakeNews-Detector

Create a virtual environment (optional but recommended):

python -m venv venv
# For Linux/Mac
source venv/bin/activate
# For Windows
venv\Scripts\activate
