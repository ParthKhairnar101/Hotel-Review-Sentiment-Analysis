# 🏨 Hotel Review Sentiment Analysis

A Machine Learning and Deep Learning project to classify hotel reviews as **positive** or **negative**, featuring a novel approach using **RoBERTa with BiLSTM** alongside traditional models like **Logistic Regression**, **XGBoost**, and **vanilla LSTM/CNN models**.

---

## 🚀 Features

- ✅ Preprocessing of real-world hotel review data
- ✅ Sentiment classification using:
  - RoBERTa + BiLSTM (custom PyTorch architecture)
  - Logistic Regression (TF-IDF / CountVectorizer)
  - XGBoost
  - LSTM, CNN (baseline models)
- ✅ Balanced training with oversampling
- ✅ Evaluation with accuracy, F1-score, confusion matrix
- ✅ TPU & GPU-compatible training loops
- ✅ Exportable models (`.pth` / `.pkl` / `.json`)

---

## 🧠 Model Architecture (Main)

```txt
Input Text → RoBERTa → BiLSTM → Dropout → Linear → Sentiment
```

This architecture enhances contextual representation (RoBERTa) with sequential dependency modeling (LSTM), setting it apart from standard transformer classifiers.

---

## 📁 Project Structure

```
├── data/                  # Input CSVs or preprocessed data
├── models/                # Saved models (.pth, .pkl, .json)
├── notebooks/             # Jupyter/Colab notebooks
├── utils/                 # Helper scripts (preprocessing, plotting, etc.)
├── README.md              # You're here!
```

---

## 📊 Sample Results

| Model                 | Accuracy | F1-Score |
|----------------------|----------|----------|
| Logistic Regression  | 78.3%    | 0.77     |
| XGBoost              | 80.1%    | 0.79     |
| LSTM                 | 81.4%    | 0.80     |
| **RoBERTa + BiLSTM** | **84.2%**| **0.83** |

> Final model trained on TPU with 2-layer BiLSTM (hidden size: 2048)

---

## 🧪 Requirements

- Python 3.9+
- PyTorch / Torch-XLA (for TPU)
- scikit-learn
- Transformers
- XGBoost
- tqdm, seaborn, matplotlib

Install requirements:
```bash
pip install -r requirements.txt
```

---

## 🧠 Dataset

- Source: [Hotel Reviews Dataset](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe/data)
- Cleaned, balanced, and merged into `Total_Review` column

---

## 🔮 Future Work

- Aspect-based Sentiment Analysis (ABSA)
- Explainability with SHAP or LIME
- Web interface for live prediction

---

## 📜 License

This project is for academic purposes only.

---

## 🙏 Acknowledgments

Special thanks to:
- HuggingFace for RoBERTa
- Google Colab TPU team
- My own patience during endless model loading 😅

---
