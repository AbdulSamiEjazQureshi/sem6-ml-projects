import unittest

from crisislens.api.server import enrich_prediction, lexical_baseline


class CrisisLensExplainTests(unittest.TestCase):
    def test_lexical_baseline_scores_keyword_rules(self):
        result = lexical_baseline("wildfire smoke evacuation near homes")

        self.assertEqual(result["label"], "fire")
        self.assertIn("wildfire", result["matched_terms"])
        self.assertGreater(result["score"], 0)

    def test_enrich_prediction_adds_research_and_response_context(self):
        model_result = {
            "label": "fire",
            "confidence": 0.82,
            "evidence": ["wildfire", "smoke"],
            "scores": {"fire": -1.2, "flood": -5.0},
        }
        challenger_result = {"label": "fire", "confidence": 1.0, "evidence": ["smoke"]}

        enriched = enrich_prediction("wildfire smoke evacuation near homes", model_result, challenger_result)

        self.assertEqual(enriched["severity"]["level"], "high")
        self.assertTrue(enriched["model_agreement"])
        self.assertIn("Notify fire response team", enriched["recommended_actions"])
        self.assertIn("not just word matching", enriched["research_note"].lower())


if __name__ == "__main__":
    unittest.main()
