"""Detect whether a news article is real or fake."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DEFAULT_MODEL = Path("fake_news_model.pkl")

SAMPLE_NEWS = pd.DataFrame(
    [
        {"label": "real", "title": "Local council approves new public library", "text": "The city council approved funding for a public library after a recorded vote on Tuesday."},
        {"label": "real", "title": "Scientists publish climate study", "text": "Researchers published their findings in a peer reviewed journal after collecting data for five years."},
        {"label": "real", "title": "School district releases annual report", "text": "The district released its annual report with enrollment and budget figures on its official website."},
        {"label": "fake", "title": "Scientists discover miracle fruit", "text": "A secret fruit instantly cures every disease and doctors do not want the public to know about it."},
        {"label": "fake", "title": "Government bans all mobile phones tomorrow", "text": "Share this shocking story now because officials are hiding the immediate nationwide phone ban."},
        {"label": "fake", "title": "Celebrity reveals impossible election result", "text": "An anonymous insider claims an unbelievable result without documents, sources, or official confirmation."},
    ]
)


def clean_text(value: object) -> str:
    text = str(value) if value is not None else ""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def combine_text(data: pd.DataFrame) -> pd.Series:
    title = data["title"] if "title" in data.columns else pd.Series("", index=data.index)
    content_column = next((name for name in ("text", "content", "article") if name in data.columns), None)
    content = data[content_column] if content_column else pd.Series("", index=data.index)
    return (title.fillna("").astype(str) + " " + content.fillna("").astype(str)).map(clean_text)


def normalize_label(value: object) -> str:
    label = str(value).strip().lower()
    if label in {"fake", "false", "0", "misleading"}:
        return "fake"
    if label in {"real", "true", "1", "reliable"}:
        return "real"
    raise ValueError(f"Unsupported label {value!r}; use fake or real.")


def load_dataset(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    label_column = next((name for name in ("label", "class", "target") if name in data.columns), None)
    if label_column is None:
        raise ValueError("CSV must contain a label, class, or target column.")
    data = data.copy()
    data["text"] = combine_text(data)
    data["label"] = data[label_column].map(normalize_label)
    data = data[data["text"].str.len() > 0]
    if data["label"].nunique() < 2:
        raise ValueError("Dataset must contain both real and fake articles.")
    return data[["text", "label"]]


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_model(data: pd.DataFrame) -> Pipeline:
    model = build_model()
    model.fit(data["text"], data["label"])
    return model


def save_model(model: Pipeline, path: str | Path) -> None:
    with Path(path).open("wb") as file:
        pickle.dump(model, file)


def load_model(path: str | Path) -> Pipeline:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def train_command(dataset: str | None, model_path: str) -> None:
    data = load_dataset(dataset) if dataset else SAMPLE_NEWS.assign(text=combine_text(SAMPLE_NEWS))[["text", "label"]]
    save_model(train_model(data), model_path)
    print(f"Trained on {len(data)} articles. Model saved to {model_path}")


def predict_command(article: str, model_path: str) -> None:
    model = load_model(model_path)
    prediction = model.predict([clean_text(article)])[0]
    confidence = max(model.predict_proba([clean_text(article)])[0])
    print(f"Prediction: {prediction.upper()} (confidence: {confidence:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify news articles as real or fake.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", help="CSV with label and title/text or content columns.")
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("article", help="News title and article text.")
    predict_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()
    if args.command == "train":
        train_command(args.dataset, args.model)
    else:
        predict_command(args.article, args.model)


if __name__ == "__main__":
    main()
