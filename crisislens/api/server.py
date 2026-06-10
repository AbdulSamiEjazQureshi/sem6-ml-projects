import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from crisislens.ml.models import MultinomialNaiveBayes, PerceptronText
from crisislens.ml.text import tokenize


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.json"

KEYWORD_RULES = {
    "fire": {"wildfire", "fire", "flames", "smoke", "evacuation", "burning"},
    "flood": {"flood", "flooded", "water", "rain", "river", "boats", "underwater"},
    "earthquake": {"earthquake", "quake", "tremor", "aftershock", "rubble", "collapse"},
    "medical": {"hospital", "medicine", "doctors", "clinic", "injured", "ambulance", "blood"},
    "relief": {"food", "blankets", "shelter", "donation", "meals", "supplies", "families"},
}

RESPONSE_ACTIONS = {
    "fire": ["Notify fire response team", "Prepare evacuation corridor", "Watch smoke inhalation reports"],
    "flood": ["Dispatch rescue boats", "Mark blocked roads", "Move families to higher shelter"],
    "earthquake": ["Inspect buildings and bridges", "Prioritize trapped-person rescue", "Prepare aftershock warning"],
    "medical": ["Route ambulances", "Request doctors and blood donors", "Track medicine shortages"],
    "relief": ["Open supply distribution point", "Prioritize food water and blankets", "Track displaced families"],
}


def load_payload():
    if not MODEL_PATH.exists():
        from crisislens.ml.train import train

        train()
    with open(MODEL_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def lexical_baseline(text):
    tokens = set(tokenize(text))
    scores = {
        label: len(tokens & keywords) for label, keywords in KEYWORD_RULES.items()
    }
    best = max(scores, key=scores.get)
    return {
        "label": best if scores[best] else "unknown",
        "score": scores[best],
        "matched_terms": sorted(tokens & KEYWORD_RULES[best]),
        "all_scores": scores,
    }


def enrich_prediction(text, model_result, challenger_result):
    baseline = lexical_baseline(text)
    confidence = model_result.get("confidence", 0.0)
    emergency_terms = len(set(tokenize(text)) & set().union(*KEYWORD_RULES.values()))
    level = "high" if confidence >= 0.75 and emergency_terms >= 2 else "medium" if emergency_terms else "low"
    label = model_result["label"]
    agreement = challenger_result["label"] == label
    return {
        **model_result,
        "lexical_baseline": baseline,
        "challenger": challenger_result,
        "model_agreement": agreement,
        "severity": {
            "level": level,
            "score": round((confidence * 70) + min(emergency_terms, 5) * 6, 2),
            "reason": f"{emergency_terms} emergency terms plus {round(confidence * 100)}% trained-model confidence",
        },
        "recommended_actions": RESPONSE_ACTIONS.get(label, ["Send to human analyst"]),
        "research_note": (
            "This is not just word matching: the lexical baseline is shown separately, "
            "while the trained Naive Bayes model uses learned class likelihoods and is compared with a perceptron challenger."
        ),
    }


class CrisisLensHandler(BaseHTTPRequestHandler):
    payload = None
    model = None
    challenger = None

    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(200, {"ok": True})

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "project": "CrisisLens"})
        elif self.path == "/experiments":
            self._send(200, self.payload["experiments"])
        elif self.path == "/clusters":
            self._send(200, self.payload["clusters"])
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/predict":
            text = body.get("text", "")
            result = enrich_prediction(text, self.model.predict_one(text), self.challenger.predict_one(text))
            self._send(200, result)
        elif self.path == "/batch":
            texts = body.get("texts", [])
            self._send(
                200,
                [
                    {
                        "text": text,
                        **enrich_prediction(text, self.model.predict_one(text), self.challenger.predict_one(text)),
                    }
                    for text in texts
                ],
            )
        else:
            self._send(404, {"error": "not found"})


def run(host="127.0.0.1", port=8011):
    payload = load_payload()
    CrisisLensHandler.payload = payload
    CrisisLensHandler.model = MultinomialNaiveBayes.from_dict(payload["primary"])
    CrisisLensHandler.challenger = PerceptronText.from_dict(payload["challenger"])
    server = ThreadingHTTPServer((host, port), CrisisLensHandler)
    print(f"CrisisLens API running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
