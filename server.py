"""
server.py — Unified Flask server for both original classifier and ABSA

Endpoints:
  POST /predict   → original sentiment + authenticity   (from predict.py)
  POST /absa      → aspect-based sentiment analysis     (from absa_predict.py)
  POST /full      → both combined in a single call
  GET  /health    → status check for both models

Usage:
  python server.py

  Optional flags:
  python server.py --port 5000
  python server.py --model muril_review_model --absa-model muril_absa_model

The web UI (index.html) talks to this server.
You only need ONE terminal to run everything.

Requirements:
  pip install flask flask-cors transformers torch
"""

import argparse
import sys
import os

# ── Lazy imports — models are loaded once at startup ──────────────────────────
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Flask not found. Install it: pip install flask flask-cors")
    sys.exit(1)


def create_app(model_dir: str, absa_model_dir: str) -> Flask:
    app = Flask(__name__)
    CORS(app)

    # ── Load models ────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  Loading models — please wait...")
    print("═" * 55)

    # Original sentiment + authenticity predictor
    try:
        from predict import ReviewPredictor
        review_predictor = ReviewPredictor(model_dir)
        review_ready     = True
        print(f"  ✓ Sentiment model loaded from '{model_dir}'")
    except FileNotFoundError:
        review_predictor = None
        review_ready     = False
        print(f"  ✗ Sentiment model NOT found at '{model_dir}'")
        print(f"    → Run train.py first to generate it.")

    # ABSA predictor
    try:
        from absa_predict import ABSAPredictor
        absa_predictor = ABSAPredictor(absa_model_dir)
        absa_ready     = absa_predictor.ml_ready
        print(
            f"  ✓ ABSA model loaded from '{absa_model_dir}'"
            if absa_ready
            else f"  ⚠ ABSA running in rule-based mode (run absa_train.py for ML mode)"
        )
    except Exception as e:
        absa_predictor = None
        absa_ready     = False
        print(f"  ✗ ABSA model failed to load: {e}")

    print("═" * 55 + "\n")

    # ── Helper ─────────────────────────────────────────────────────────────────
    def parse_texts(data: dict) -> list[str] | None:
        texts = data.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]
        return texts if texts else None

    # ── Routes ─────────────────────────────────────────────────────────────────

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status":         "ok",
            "sentiment_model": "ready" if review_ready else "not loaded",
            "absa_model":      "ml"    if absa_ready  else ("rule-based" if absa_predictor else "not loaded"),
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        """Original sentiment + authenticity prediction."""
        if not review_ready:
            return jsonify({
                "error": "Sentiment model not loaded. Run train.py first."
            }), 503

        data  = request.get_json(force=True)
        texts = parse_texts(data)
        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        try:
            results = review_predictor.predict(texts)
            return jsonify({"results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/absa", methods=["POST"])
    def absa():
        """Aspect-based sentiment analysis."""
        if not absa_predictor:
            return jsonify({
                "error": "ABSA model not loaded. Run absa_train.py first."
            }), 503

        data  = request.get_json(force=True)
        texts = parse_texts(data)
        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        try:
            results = absa_predictor.predict(texts)
            return jsonify({"results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/full", methods=["POST"])
    def full():
        """
        Combined endpoint — returns both sentiment+authenticity AND aspect analysis
        in a single API call. Used by the unified index.html.

        Response shape:
        {
          "results": [
            {
              "text": "...",

              // From predict.py
              "sentiment": "positive",
              "sentiment_confidence": 91.2,
              "sentiment_scores": {...},
              "authenticity": "genuine",
              "authenticity_confidence": 88.5,
              "authenticity_scores": {...},

              // From absa_predict.py
              "aspects_detected": ["delivery", "product_quality"],
              "aspect_sentiments": {
                "delivery": {"label": "positive", "confidence": 87.0, "source": "ml"},
                ...
              },
              "absa_summary": "Delivery: 👍 positive (87%) | Product Quality: 👎 negative (79%)"
            }
          ]
        }
        """
        data  = request.get_json(force=True)
        texts = parse_texts(data)
        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        combined = [{
            "text": t,
            # Sentiment defaults
            "sentiment": "unavailable",
            "sentiment_confidence": 0,
            "sentiment_scores": {},
            "authenticity": "unavailable",
            "authenticity_confidence": 0,
            "authenticity_scores": {},
            # ABSA defaults
            "aspects_detected": [],
            "aspect_sentiments": {},
            "absa_summary": "ABSA unavailable",
        } for t in texts]

        # Merge sentiment results
        if review_ready:
            try:
                sent_results = review_predictor.predict(texts)
                for i, r in enumerate(sent_results):
                    combined[i].update({
                        "sentiment":                r["sentiment"],
                        "sentiment_confidence":     r["sentiment_confidence"],
                        "sentiment_scores":         r["sentiment_scores"],
                        "authenticity":             r["authenticity"],
                        "authenticity_confidence":  r["authenticity_confidence"],
                        "authenticity_scores":      r["authenticity_scores"],
                    })
            except Exception as e:
                for item in combined:
                    item["sentiment_error"] = str(e)

        # Merge ABSA results
        if absa_predictor:
            try:
                absa_results = absa_predictor.predict(texts)
                for i, r in enumerate(absa_results):
                    combined[i].update({
                        "aspects_detected":  r["aspects_detected"],
                        "aspect_sentiments": r["aspect_sentiments"],
                        "absa_summary":      r["absa_summary"],
                    })
            except Exception as e:
                for item in combined:
                    item["absa_error"] = str(e)

        return jsonify({"results": combined})

    return app


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Unified server for sentiment + ABSA models"
    )
    parser.add_argument("--model",      default="muril_review_model",
                        help="Directory of original sentiment model")
    parser.add_argument("--absa-model", default="muril_absa_model",
                        help="Directory of ABSA model")
    parser.add_argument("--port",       type=int, default=5000,
                        help="Port to run server on (default: 5000)")
    parser.add_argument("--host",       default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    app = create_app(args.model, args.absa_model)

    print(f"  Server   : http://localhost:{args.port}")
    print(f"  Endpoints:")
    print(f"    GET  /health   — model status")
    print(f"    POST /predict  — sentiment + authenticity")
    print(f"    POST /absa     — aspect-based sentiment")
    print(f"    POST /full     — both combined")
    print(f"\n  Open index.html in your browser to use the UI\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
