import unittest

from crisislens.ml.metrics import accuracy, confusion_matrix, macro_f1
from crisislens.ml.models import KMeansText, MultinomialNaiveBayes, PerceptronText
from crisislens.ml.text import TfidfVectorizer, tokenize


class CrisisLensCoreTests(unittest.TestCase):
    def test_tokenize_removes_noise_and_stopwords(self):
        self.assertEqual(
            tokenize("URGENT: Flooding near bridge!!! Need rescue now."),
            ["urgent", "flooding", "near", "bridge", "need", "rescue"],
        )

    def test_naive_bayes_predicts_seen_crisis_language(self):
        texts = [
            "river flood rescue boats needed",
            "homes under water after heavy rain",
            "wildfire smoke evacuation order",
            "forest fire spreading fast",
        ]
        labels = ["flood", "flood", "fire", "fire"]
        model = MultinomialNaiveBayes()
        model.fit(texts, labels)

        prediction = model.predict_one("evacuation as wildfire smoke spreads")

        self.assertEqual(prediction["label"], "fire")
        self.assertGreater(prediction["confidence"], 0.5)
        self.assertIn("wildfire", prediction["evidence"])

    def test_perceptron_learns_separable_short_text(self):
        texts = [
            "quake tremor buildings cracked",
            "aftershock damaged roads",
            "hospital needs medicine doctors",
            "clinic requires blood supplies",
        ]
        labels = ["earthquake", "earthquake", "medical", "medical"]
        model = PerceptronText(epochs=8)
        model.fit(texts, labels)

        prediction = model.predict_one("doctors need medicine at clinic")

        self.assertEqual(prediction["label"], "medical")

    def test_metrics_report_macro_f1_and_confusion_matrix(self):
        truth = ["flood", "flood", "fire", "medical"]
        pred = ["flood", "fire", "fire", "medical"]

        self.assertEqual(accuracy(truth, pred), 0.75)
        self.assertGreater(macro_f1(truth, pred), 0.7)
        self.assertEqual(confusion_matrix(truth, pred)["flood"]["fire"], 1)

    def test_text_clustering_groups_related_messages(self):
        texts = [
            "river flood water rescue",
            "boats rescue families from flood",
            "wildfire smoke evacuation",
            "forest fire smoke spreading",
        ]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        clusters = KMeansText(k=2, iterations=6).fit_predict(matrix)

        self.assertEqual(clusters[0], clusters[1])
        self.assertEqual(clusters[2], clusters[3])
        self.assertNotEqual(clusters[0], clusters[2])


if __name__ == "__main__":
    unittest.main()
