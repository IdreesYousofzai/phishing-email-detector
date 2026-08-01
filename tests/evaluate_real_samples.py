"""
evaluate_real_samples.py

The build plan called for testing against real phishing samples from the
PhishTank public archive. PhishTank's archive is reached through a live
API/site fetch that this environment cannot reach directly, so this script
instead does the equivalent job on real, previously-unseen phishing and
legitimate emails: a held-out slice of the Nazario phishing corpus and the
Enron legitimate corpus that was set aside and NEVER used in training.

This is the same measurement the plan asked for - false positive rate and
false negative rate on genuine, real-world messages the model has not
memorised - just sourced from a corpus that's reachable offline.

To point this at PhishTank instead: download the current verified_online.csv
export from https://phishtank.org/developer_info.php (needs an API key) and
adapt `load_holdout_samples()` to read it into the same subject/body/label
shape.
"""

import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import analyse_email  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent



def load_holdout_samples(n_per_class=150):
    """Pull emails NOT present in the training dataset (data/phishing_email_dataset.csv)
    from the original raw corpora, so this is a genuine held-out real-world test."""
    train_df = pd.read_csv(BASE_DIR / "data" / "phishing_email_dataset.csv")
    used_bodies = set(train_df["body"].astype(str))

    raw_dir = BASE_DIR.parent  # where Nazario.csv / Enron.csv were downloaded
    nazario = pd.read_csv(raw_dir / "Nazario.csv", encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    enron = pd.read_csv(raw_dir / "Enron.csv", encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")

    nazario = nazario[~nazario["body"].astype(str).isin(used_bodies)]
    enron = enron[~enron["body"].astype(str).isin(used_bodies)]

    phishing_sample = nazario.sample(n=min(n_per_class, len(nazario)), random_state=7)
    legit_sample = enron[enron["label"] == 0].sample(n=min(n_per_class, len(enron)), random_state=7)

    return phishing_sample, legit_sample



def main():
    phishing_sample, legit_sample = load_holdout_samples()
    print(f"Held-out real phishing samples: {len(phishing_sample)}")
    print(f"Held-out real legitimate samples: {len(legit_sample)}")

    false_negatives = 0  # phishing missed, called legitimate
    false_positives = 0  # legitimate wrongly flagged as phishing

    for _, row in phishing_sample.iterrows():
        result = analyse_email(subject=str(row.get("subject", "")), body=str(row["body"]))
        if result["verdict"] == "Legitimate":
            false_negatives += 1

    for _, row in legit_sample.iterrows():
        result = analyse_email(subject=str(row.get("subject", "")), body=str(row["body"]))
        if result["verdict"] == "Phishing":
            false_positives += 1

    fnr = false_negatives / len(phishing_sample) * 100
    fpr = false_positives / len(legit_sample) * 100

    print("\n=== Real-sample evaluation ===")
    print(f"False negative rate (missed phishing): {false_negatives}/{len(phishing_sample)} = {fnr:.1f}%")
    print(f"False positive rate (legit flagged):    {false_positives}/{len(legit_sample)} = {fpr:.1f}%")
    print(
        "\nA higher false positive rate is generally preferable to a higher false negative "
        "rate here - a wrongly-flagged legitimate email costs the user a moment's confusion, "
        "while a missed phishing email can cost credentials or money."
    )


if __name__ == "__main__":
    main()
