"""
predict.py — Run inference with the trained MuRIL dual classifier

Usage (single text):
  python predict.py --text "यह प्रोडक्ट बहुत अच्छा है!"

Usage (batch CSV):
  python predict.py --csv test_data.csv --output predictions.csv

Usage (Flask API server — used by the web UI):
  python predict.py --server
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ── Reproduce the same model class as train.py ────────────────────────────────
class MuRILDualClassifier(nn.Module):
    def __init__(self, model_name, num_sentiment, num_fake, dropout=0.3):
        super().__init__()
        self.encoder        = AutoModel.from_pretrained(model_name)
        hidden              = self.encoder.config.hidden_size
        self.dropout        = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden, num_sentiment)
        self.fake_head      = nn.Linear(hidden, num_fake)

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        return self.sentiment_head(pooled), self.fake_head(pooled)


# ── Predictor class ────────────────────────────────────────────────────────────
class ReviewPredictor:
    def __init__(self, model_dir="muril_review_model", max_len=128):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len

        label_path = os.path.join(model_dir, "label_map.json")
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"label_map.json not found in {model_dir}. "
                "Please run train.py first."
            )

        with open(label_path) as f:
            label_map = json.load(f)

        self.sentiment_labels = label_map["sentiment_labels"]
        self.fake_labels      = label_map["fake_labels"]
        base_model            = label_map["base_model"]

        print(f"Loading tokenizer from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        print(f"Loading model weights from {model_dir}/model.pt...")
        self.model = MuRILDualClassifier(
            base_model,
            num_sentiment=len(self.sentiment_labels),
            num_fake=len(self.fake_labels),
        )
        state = torch.load(
            os.path.join(model_dir, "model.pt"),
            map_location=self.device,
            weights_only=True
        )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print("Model ready.")

    def predict(self, texts: list[str]) -> list[dict]:
        enc = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            sent_logits, fake_logits = self.model(input_ids, attention_mask)

        sent_probs = F.softmax(sent_logits, dim=1).cpu().numpy()
        fake_probs = F.softmax(fake_logits, dim=1).cpu().numpy()

        results = []
        for i, text in enumerate(texts):
            sent_idx  = sent_probs[i].argmax()
            fake_idx  = fake_probs[i].argmax()
            results.append({
                "text":                 text,
                "sentiment":            self.sentiment_labels[sent_idx],
                "sentiment_confidence": round(float(sent_probs[i][sent_idx]) * 100, 1),
                "sentiment_scores": {
                    label: round(float(sent_probs[i][j]) * 100, 1)
                    for j, label in enumerate(self.sentiment_labels)
                },
                "authenticity":            self.fake_labels[fake_idx],
                "authenticity_confidence": round(float(fake_probs[i][fake_idx]) * 100, 1),
                "authenticity_scores": {
                    label: round(float(fake_probs[i][j]) * 100, 1)
                    for j, label in enumerate(self.fake_labels)
                },
            })
        return results


# ── Flask server (used by web UI) ──────────────────────────────────────────────
def run_server(model_dir, port=5000):
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("Flask not found. Install it: pip install flask flask-cors")
        sys.exit(1)

    app       = Flask(__name__)
    CORS(app)
    predictor = ReviewPredictor(model_dir)

    @app.route("/predict", methods=["POST"])
    def predict():
        data  = request.get_json()
        texts = data.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        results = predictor.predict(texts)
        return jsonify({"results": results})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    print(f"\n Server running at http://localhost:{port}")
    print(f" Web UI: open index.html in your browser")
    app.run(port=port, debug=False)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="muril_review_model", help="Saved model directory")
    parser.add_argument("--text",   help="Single review text to classify")
    parser.add_argument("--csv",    help="CSV file with a 'text' column")
    parser.add_argument("--output", default="predictions.csv",    help="Output CSV for batch mode")
    parser.add_argument("--server", action="store_true",          help="Start Flask API server")
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    if args.server:
        run_server(args.model, args.port)
        return

    predictor = ReviewPredictor(args.model)

    if args.text:
        results = predictor.predict([args.text])
        r = results[0]
        print(f"\nText:        {r['text']}")
        print(f"Sentiment:   {r['sentiment'].upper()} ({r['sentiment_confidence']}%)")
        print(f"             Scores: {r['sentiment_scores']}")
        print(f"Authenticity:{r['authenticity'].upper()} ({r['authenticity_confidence']}%)")
        print(f"             Scores: {r['authenticity_scores']}")

    elif args.csv:
        df      = pd.read_csv(args.csv)
        texts   = df["text"].tolist()
        results = predictor.predict(texts)
        out_df  = pd.DataFrame(results)
        out_df.to_csv(args.output, index=False)
        print(f"Saved {len(results)} predictions to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
