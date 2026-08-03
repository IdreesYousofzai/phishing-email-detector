"""
predict.py
Combines the trained ML model with the rule-based feature checks into a
single verdict, used by both the Flask app and the offline test scripts.
"""

import json
from pathlib import Path

import joblib

from features import extract_rule_features
from preprocess import clean_text, combine_subject_body

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

_model = None
_vectorizer = None


def _load():
    global _model, _vectorizer
    if _model is None:
        _model = joblib.load(MODELS_DIR / "best_model.pkl")
        _vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
    return _model, _vectorizer




def analyse_email(subject: str, body: str, display_name: str = "", sender_email: str = "") -> dict:
    """
    Returns a full verdict dictionary:
      - verdict: "Phishing" or "Legitimate"
      - confidence: 0-100
      - ml_probability_phishing: raw model probability
      - rule_score: 0-5 rule-based score
      - triggered_reasons: list of human-readable reasons
    """
    model, vectorizer = _load()

    combined = combine_subject_body(subject, body)
    cleaned = clean_text(combined)

    X = vectorizer.transform([cleaned])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        phishing_prob = float(proba[1])
    else:
        pred = model.predict(X)[0]
        phishing_prob = float(pred)

    rules = extract_rule_features(subject, body, display_name, sender_email)

    # Blend: ML probability is the primary signal, rule_score nudges it.
    # Each triggered rule shifts the score by up to 4 percentage points,
    # capped so rules alone can't flip a confident ML verdict outright.
    rule_adjustment = min(rules["rule_score"] * 0.04, 0.18)
    if rules["rule_score"] >= 3:
        blended_prob = phishing_prob + rule_adjustment
    else:
        blended_prob = phishing_prob
    blended_prob = max(0.0, min(1.0, blended_prob))

    verdict = "Phishing" if blended_prob >= 0.5 else "Legitimate"
    confidence = blended_prob * 100 if verdict == "Phishing" else (1 - blended_prob) * 100

    return {
        "verdict": verdict,
        "confidence": round(confidence, 1),
        "ml_probability_phishing": round(phishing_prob * 100, 1),
        "rule_score": rules["rule_score"],
        "rule_score_max": rules["rule_score_max"],
        "triggered_reasons": rules["triggered_reasons"],
        "urls_found": rules["urls"]["total_urls"],
        "suspicious_urls": rules["urls"]["suspicious_urls"],
        "sender_mismatch": rules["sender_mismatch"],
    }


if __name__ == "__main__":
    result = analyse_email(
        subject="URGENT: Verify your account now",
        body=(
            "Dear Customer, we detected unusual activity on your account. "
            "Your account will be suspended within 24 hours unless you verify your account now. "
            "Click here: http://paypa1-secure-login.ru/verify"
        ),
        display_name="PayPal Support",
        sender_email="support@paypa1-secure-login.ru",
    )
    print(json.dumps(result, indent=2))
