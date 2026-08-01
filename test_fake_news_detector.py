from fake_news_detector import SAMPLE_NEWS, combine_text, load_dataset, train_model


def test_model_classifies_sample_articles():
    data = SAMPLE_NEWS.assign(text=combine_text(SAMPLE_NEWS))[["text", "label"]]
    model = train_model(data)
    assert list(model.predict(data["text"])) == list(data["label"])


def test_csv_loader_reads_real_and_fake_labels():
    data = load_dataset("sample_news.csv")
    assert len(data) == 6
    assert set(data["label"]) == {"real", "fake"}
