# Sentiment Analysis — NLP & Naive Bayes Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-87.20%25-brightgreen)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![DecodeLabs](https://img.shields.io/badge/DecodeLabs-Project%204-orange)

---

## Project Overview

This is **Project 4** of the DecodeLabs Industrial Training Kit (Batch 2026).
This project builds a production-grade NLP sentiment analysis pipeline on
the IMDB 50K Movie Reviews dataset — using text preprocessing with
negation-safe stop words, TF-IDF vectorization with bigrams, and
Multinomial Naive Bayes classification.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — IMDB Dataset of 50K Movie Reviews |
| Rows | 50,000 reviews |
| Balance | 25,000 positive / 25,000 negative |
| Target | sentiment (positive/negative) |

---

## Pipeline Architecture
Raw IMDB Reviews (50,000)
↓
STEP 1: TEXT PREPROCESSING

Remove HTML tags (<br />)
Remove special characters
Lowercase normalization
Negation-safe stop word removal
POS-guided lemmatization
↓
STEP 2: TF-IDF VECTORIZATION
Unigrams + Bigrams (ngram_range=(1,2))
Top 10,000 features
CSR sparse matrix format
↓
STEP 3: TRAIN/TEST SPLIT (80/20)
↓
STEP 4: MULTINOMIAL NAIVE BAYES
alpha=0.1 (Laplace smoothing)
↓
STEP 5: EVALUATION
Accuracy, Precision, Recall, F1
Confusion Matrix
Top positive/negative words

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | **87.20%** |
| Precision | 86.13% |
| Recall | 88.68% |
| F1 Score | **87.39%** |

---

## Confusion Matrix

| | Predicted Negative | Predicted Positive |
|--|-------------------|-------------------|
| Actual Negative | 4,286 | 714 |
| Actual Positive | 566 | 4,434 |

---

## Key Design Decisions

**Why Multinomial Naive Bayes?**
Perfectly balanced dataset (50/50) — no class imbalance to handle.
Works natively with sparse TF-IDF matrices. Fast and interpretable.

**Why negation-safe stop words?**
Standard stop word lists remove "not", "never", "don't" — destroying
critical sentiment signals. "not good" becomes "good" — catastrophic
for sentiment analysis!

**Why bigrams?**
Unigrams alone miss phrases like "not worth", "complete waste",
"highly recommended". Bigrams capture these multi-word sentiment signals.

**Why TF-IDF over Bag-of-Words?**
TF-IDF penalizes common words that appear everywhere (low signal)
and rewards rare distinctive words (high signal).

---

## Live Predictions
Review: "This movie was absolutely amazing!"
Sentiment: POSITIVE — Confidence: 67.25%
Review: "Terrible movie, complete waste of time."
Sentiment: NEGATIVE — Confidence: 99.75%
Review: "Not bad but could have been better."
Sentiment: NEGATIVE — Confidence: 71.97%

---

## Project Structure
sentiment-analysis/
│
├── data/
│   └── IMDB Dataset.csv         ← raw dataset (gitignored)
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── src/
│   ├── pipeline.py              ← end-to-end pipeline
│   ├── preprocess.py            ← text cleaning module
│   ├── train.py                 ← TF-IDF + Naive Bayes
│   └── predict.py               ← load model & predict
│
├── outputs/
│   └── sentiment_results.csv
│
├── .gitignore
└── README.md

---

## How to Run

**1. Clone:**
```bash
git clone https://github.com/mehakchanna04-coder/sentiment-analysis.git
cd sentiment-analysis
```

**2. Install:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk jupyter
```

**3. Add dataset:**
- Download `IMDB Dataset.csv` from Kaggle
- Place in `data/` folder

**4. Run full pipeline:**
```bash
python src/pipeline.py
```

**5. Predict sentiment:**
```bash
python src/predict.py
```

---

## Key Learnings

- Standard stop word lists destroy negation signals — always use
  negation-safe stop words for sentiment analysis
- Bigrams capture multi-word sentiment phrases invisible to unigrams
- TF-IDF sparse matrices (CSR format) handle 50K×10K efficiently
- MultinomialNB is ideal for balanced text classification tasks
- 87.20% accuracy on 50K reviews proves the pipeline is production-ready

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| NLTK | Tokenization, lemmatization, stop words |
| Scikit-learn | TF-IDF, Naive Bayes, metrics |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Google Colab | Development environment |
| Git + GitHub | Version control |

---

## Complete Portfolio

| Project | Repo | Topic | Result |
|---------|------|-------|--------|
| Project 1 | house-price-ds | EDA & Feature Engineering | R2=0.9066 |
| Project 2 | fraud-detection | Fraud Detection | ROC-AUC=0.98 |
| Project 3 | customer-segmentation | Customer Segmentation | 5 Personas |
| Project 4 | sentiment-analysis | NLP Sentiment Analysis | 87.20% |

---

## Author

**Mehak Channa**
DecodeLabs Industrial Training — Batch 2026
GitHub: [@mehakchanna04-coder](https://github.com/mehakchanna04-coder)

---

*"Build a pipeline that detects sentiment with absolute clarity."*
*— DecodeLabs Project 4*
