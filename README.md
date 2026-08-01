# Fake News Detector

A Python NLP project that classifies news articles as `REAL` or `FAKE` using Pandas, TF-IDF, and Logistic Regression.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

Use the included sample dataset:

```powershell
python fake_news_detector.py train --dataset sample_news.csv
```

Your CSV should contain:

- `label`, `class`, or `target`: `real`/`fake` labels
- `title`: article title (optional)
- `text`, `content`, or `article`: article body

Example with a larger dataset:

```powershell
python fake_news_detector.py train --dataset news_dataset.csv --model news_model.pkl
```

## Predict

```powershell
python fake_news_detector.py predict "Anonymous source claims a miracle cure is being hidden from everyone"
```

Use a custom saved model with `--model news_model.pkl`.

## Test

```powershell
pytest -q
```

This is an educational classifier, not a replacement for professional fact-checking. Accuracy depends on the quality, balance, language, and source coverage of the training dataset. Evaluate with a held-out test set using precision, recall, F1-score, and a confusion matrix.
