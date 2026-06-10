import math
from collections import Counter, defaultdict

from crisislens.ml.text import TfidfVectorizer, tokenize


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


class MultinomialNaiveBayes:
    def __init__(self):
        self.labels = []
        self.label_priors = {}
        self.token_log_probs = {}
        self.default_log_probs = {}
        self.vocabulary = []

    def fit(self, texts, labels):
        self.labels = sorted(set(labels))
        label_counts = Counter(labels)
        token_counts = {label: Counter() for label in self.labels}
        vocabulary = set()
        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            token_counts[label].update(tokens)
            vocabulary.update(tokens)
        self.vocabulary = sorted(vocabulary)
        total_docs = len(labels)
        for label in self.labels:
            self.label_priors[label] = math.log(label_counts[label] / total_docs)
            denominator = sum(token_counts[label].values()) + len(self.vocabulary)
            self.default_log_probs[label] = math.log(1 / denominator)
            self.token_log_probs[label] = {
                token: math.log((token_counts[label][token] + 1) / denominator)
                for token in self.vocabulary
            }
        return self

    def predict_one(self, text):
        tokens = tokenize(text)
        scores = {}
        evidence = {}
        for label in self.labels:
            score = self.label_priors[label]
            token_scores = {}
            for token in tokens:
                contribution = self.token_log_probs[label].get(
                    token, self.default_log_probs[label]
                )
                score += contribution
                token_scores[token] = contribution
            scores[label] = score
            evidence[label] = sorted(token_scores, key=token_scores.get, reverse=True)[:5]
        best = max(scores, key=scores.get)
        shifted = {label: math.exp(value - max(scores.values())) for label, value in scores.items()}
        total = sum(shifted.values()) or 1.0
        return {
            "label": best,
            "confidence": shifted[best] / total,
            "scores": scores,
            "evidence": evidence[best],
        }

    def predict(self, texts):
        return [self.predict_one(text)["label"] for text in texts]

    def to_dict(self):
        return {
            "kind": "multinomial_naive_bayes",
            "labels": self.labels,
            "label_priors": self.label_priors,
            "token_log_probs": self.token_log_probs,
            "default_log_probs": self.default_log_probs,
            "vocabulary": self.vocabulary,
        }

    @classmethod
    def from_dict(cls, payload):
        model = cls()
        model.labels = payload["labels"]
        model.label_priors = payload["label_priors"]
        model.token_log_probs = payload["token_log_probs"]
        model.default_log_probs = payload["default_log_probs"]
        model.vocabulary = payload["vocabulary"]
        return model


class PerceptronText:
    def __init__(self, epochs=6):
        self.epochs = epochs
        self.vectorizer = TfidfVectorizer()
        self.labels = []
        self.weights = {}

    def fit(self, texts, labels):
        matrix = self.vectorizer.fit_transform(texts)
        self.labels = sorted(set(labels))
        self.weights = {label: [0.0] * len(self.vectorizer.vocabulary) for label in self.labels}
        for _ in range(self.epochs):
            for vector, expected in zip(matrix, labels):
                actual = self._predict_vector(vector)
                if actual != expected:
                    for index, value in enumerate(vector):
                        self.weights[expected][index] += value
                        self.weights[actual][index] -= value
        return self

    def _predict_vector(self, vector):
        scores = {label: _dot(weights, vector) for label, weights in self.weights.items()}
        return max(scores, key=scores.get)

    def predict_one(self, text):
        vector = self.vectorizer.transform([text])[0]
        scores = {label: _dot(weights, vector) for label, weights in self.weights.items()}
        best = max(scores, key=scores.get)
        ranked = sorted(range(len(vector)), key=lambda idx: vector[idx], reverse=True)
        inverse_vocab = {index: token for token, index in self.vectorizer.vocabulary.items()}
        evidence = [inverse_vocab[index] for index in ranked[:5] if vector[index] > 0]
        return {"label": best, "confidence": 1.0, "scores": scores, "evidence": evidence}

    def predict(self, texts):
        return [self.predict_one(text)["label"] for text in texts]

    def to_dict(self):
        return {
            "kind": "perceptron",
            "epochs": self.epochs,
            "vectorizer": self.vectorizer.to_dict(),
            "labels": self.labels,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, payload):
        model = cls(payload["epochs"])
        model.vectorizer = TfidfVectorizer.from_dict(payload["vectorizer"])
        model.labels = payload["labels"]
        model.weights = payload["weights"]
        return model


class KMeansText:
    def __init__(self, k=3, iterations=10):
        self.k = k
        self.iterations = iterations
        self.centroids = []

    def fit_predict(self, matrix):
        if not matrix:
            return []
        self.centroids = [matrix[0][:]]
        while len(self.centroids) < self.k and len(self.centroids) < len(matrix):
            farthest = max(
                matrix,
                key=lambda row: min(_distance(row, centroid) for centroid in self.centroids),
            )
            self.centroids.append(farthest[:])
        assignments = [0] * len(matrix)
        for _ in range(self.iterations):
            assignments = [
                min(range(len(self.centroids)), key=lambda idx: _distance(row, self.centroids[idx]))
                for row in matrix
            ]
            grouped = defaultdict(list)
            for assignment, row in zip(assignments, matrix):
                grouped[assignment].append(row)
            for idx in range(len(self.centroids)):
                rows = grouped.get(idx)
                if not rows:
                    continue
                self.centroids[idx] = [
                    sum(row[col] for row in rows) / len(rows) for col in range(len(rows[0]))
                ]
        return assignments
