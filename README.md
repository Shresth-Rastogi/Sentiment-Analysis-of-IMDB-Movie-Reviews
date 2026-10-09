# IMDB Movie Review Sentiment Analysis

An end-to-end Natural Language Processing (NLP) pipeline for binary sentiment classification on the IMDB 50K movie reviews dataset. The project covers exploratory data analysis (EDA), data cleaning and text normalization, n-gram analysis, and sentiment classification using scikit-learn.

---

## Project Overview

- **Dataset**: IMDB Dataset of 50,000 Movie Reviews
- **Target**: Binary classification (`positive` vs. `negative`)
- **Initial Shape**: 50,000 rows × 2 columns (`review`, `sentiment`)
- **Duplicate Cleaned Shape**: 49,582 reviews (418 exact duplicate reviews removed)
- **Techniques Used**:
  - Exploratory Data Analysis & Text Diagnostics
  - HTML stripping via Regular Expressions
  - Tokenization, n-gram extraction, and log-odds ratio discriminative word analysis
  - Vectorization (`CountVectorizer`, `TfidfVectorizer`)
  - Classification using `LogisticRegression`
  - Model evaluation with Confusion Matrix, Classification Report, and ROC-AUC

---

## Project Structure

```text
├── IMDB_dataset.csv            # Dataset containing review text and sentiment labels
├── nlp_project_final.ipynb     # Jupyter Notebook containing EDA, preprocessing, and modeling
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
