import csv
import json
from pathlib import Path

from crisislens.ml.metrics import accuracy, confusion_matrix, macro_f1, precision_recall_f1
from crisislens.ml.models import KMeansText, MultinomialNaiveBayes, PerceptronText
from crisislens.ml.text import TfidfVectorizer, tokenize


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tweets.csv"
MODEL_PATH = ROOT / "models" / "model.json"
REPORT_DIR = ROOT / "report"


def load_dataset(path=DATA_PATH):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row["text"] for row in rows], [row["label"] for row in rows]


def split_dataset(texts, labels):
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    by_label = {}
    for index, (text, label) in enumerate(zip(texts, labels)):
        by_label.setdefault(label, []).append(index)
    test_indexes = {indexes[-1] for indexes in by_label.values()}
    for index, (text, label) in enumerate(zip(texts, labels)):
        if index in test_indexes:
            test_texts.append(text)
            test_labels.append(label)
        else:
            train_texts.append(text)
            train_labels.append(label)
    return train_texts, train_labels, test_texts, test_labels


def evaluate(name, model, test_texts, test_labels):
    predictions = model.predict(test_texts)
    return {
        "model": name,
        "accuracy": round(accuracy(test_labels, predictions), 4),
        "macro_f1": round(macro_f1(test_labels, predictions), 4),
        "per_class": precision_recall_f1(test_labels, predictions),
        "confusion_matrix": confusion_matrix(test_labels, predictions),
        "predictions": predictions,
    }


def cluster_summary(texts):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    assignments = KMeansText(k=4, iterations=8).fit_predict(matrix)
    clusters = {}
    for assignment, text in zip(assignments, texts):
        clusters.setdefault(str(assignment), []).append(text)
    summary = []
    for cluster_id, rows in clusters.items():
        counts = {}
        for text in rows:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        keywords = [token for token, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:6]]
        summary.append({"cluster": cluster_id, "size": len(rows), "keywords": keywords, "examples": rows[:3]})
    return summary


def write_reports(results, clusters):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "experiments.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "accuracy", "macro_f1"])
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in ["model", "accuracy", "macro_f1"]})
    with open(REPORT_DIR / "confusion_matrix.json", "w", encoding="utf-8") as handle:
        json.dump({result["model"]: result["confusion_matrix"] for result in results}, handle, indent=2)
    with open(REPORT_DIR / "clusters.json", "w", encoding="utf-8") as handle:
        json.dump(clusters, handle, indent=2)


def train():
    texts, labels = load_dataset()
    train_texts, train_labels, test_texts, test_labels = split_dataset(texts, labels)
    nb = MultinomialNaiveBayes().fit(train_texts, train_labels)
    perceptron = PerceptronText(epochs=10).fit(train_texts, train_labels)
    results = [
        evaluate("Multinomial Naive Bayes", nb, test_texts, test_labels),
        evaluate("Linear Perceptron", perceptron, test_texts, test_labels),
    ]
    clusters = cluster_summary(texts)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "primary": nb.to_dict(),
                "challenger": perceptron.to_dict(),
                "experiments": results,
                "clusters": clusters,
            },
            handle,
            indent=2,
        )
    write_reports(results, clusters)
    return results


if __name__ == "__main__":
    for row in train():
        print(f"{row['model']}: accuracy={row['accuracy']} macro_f1={row['macro_f1']}")
