"""
app.py
Flask web interface for the phishing email detector.

Paste an email (optionally with sender display name and sender address),
click Analyse, and see the verdict, confidence percentage and the exact
features that drove the classification.
"""

import json
import sys
from pathlib import Path

from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import analyse_email  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {"subject": "", "body": "", "display_name": "", "sender_email": ""}

    if request.method == "POST":
        form_data["subject"] = request.form.get("subject", "")
        form_data["body"] = request.form.get("body", "")
        form_data["display_name"] = request.form.get("display_name", "")
        form_data["sender_email"] = request.form.get("sender_email", "")

        if form_data["body"].strip():
            result = analyse_email(
                subject=form_data["subject"],
                body=form_data["body"],
                display_name=form_data["display_name"],
                sender_email=form_data["sender_email"],
            )

    metrics = {}
    metrics_path = BASE_DIR / "models" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return render_template("index.html", result=result, form_data=form_data, metrics=metrics)


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """JSON API endpoint, in case someone wants to hit this programmatically."""
    data = request.get_json(force=True) or {}
    result = analyse_email(
        subject=data.get("subject", ""),
        body=data.get("body", ""),
        display_name=data.get("display_name", ""),
        sender_email=data.get("sender_email", ""),
    )
    return result


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
