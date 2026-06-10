import math
import re
from collections import Counter


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "for",
    "from",
    "in",
    "is",
    "now",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z_'-]*", text.lower())
    return [word.strip("'") for word in words if word not in STOPWORDS and len(word) > 1]


class TfidfVectorizer:
    def __init__(self, vocabulary=None, idf=None):
        self.vocabulary = vocabulary or {}
        self.idf = idf or {}

    def fit(self, texts):
        document_frequency = Counter()
        for text in texts:
            document_frequency.update(set(tokenize(text)))
        self.vocabulary = {
            token: index for index, token in enumerate(sorted(document_frequency))
        }
        total = max(len(texts), 1)
        self.idf = {
            token: math.log((1 + total) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }
        return self

    def transform(self, texts):
        vectors = []
        for text in texts:
            counts = Counter(tokenize(text))
            total = sum(counts.values()) or 1
            vector = [0.0] * len(self.vocabulary)
            for token, count in counts.items():
                if token in self.vocabulary:
                    vector[self.vocabulary[token]] = (count / total) * self.idf.get(token, 1.0)
            vectors.append(vector)
        return vectors

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def to_dict(self):
        return {"vocabulary": self.vocabulary, "idf": self.idf}

    @classmethod
    def from_dict(cls, payload):
        return cls(payload["vocabulary"], payload["idf"])
