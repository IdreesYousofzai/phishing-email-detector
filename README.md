# Phishing Email Detector

A phishing email classifier that combines a trained ML model with rule-based
checks (urgency language, suspicious URLs, sender/domain mismatch), served
through a Flask web app.

Paste an email in, get a verdict, a confidence percentage, and the exact
reasons why it was flagged.

<img width="890" height="406" alt="image" src="https://github.com/user-attachments/assets/dbb01fc8-1181-4a82-bdf9-acde5c82138d" />


## Why this matters

Phishing is the entry point for most breaches. Verizon's Data Breach
Investigations Report has repeatedly found that the majority of confirmed
breaches involve some form of social engineering or credential theft that
starts with a phishing email. A single convincing message is often the
whole attack: no exploit needed, just someone clicking a link and typing
their password into the wrong page.

## Dataset

The build plan called for a Kaggle phishing dataset. This environment
can't reach Kaggle directly, so I pulled from three public, labelled email
corpora instead (Nazario phishing corpus, SpamAssassin, and Enron), which
between them cover the same ground: real phishing emails and real
legitimate business email.

- Combined: 37,003 emails after deduplication and cleaning
- Sampled down to 8,000 for training (3,200 phishing / 4,800 legitimate),
  keeping a realistic imbalance rather than an artificial 50/50 split
- Columns used: `subject`, `body`, `label` (1 = phishing, 0 = legitimate)

## Preprocessing

Each email's subject and body are combined (subject counted twice, since
it's short but carries a lot of the urgency signal), then cleaned:

1. Lowercase everything
2. Strip HTML tags
3. Strip raw URLs (handled separately by the rule-based URL checker)
4. Remove punctuation and digits
5. Tokenise on whitespace
6. Drop stop words and single-character tokens

## Feature extraction

TF-IDF vectorisation, unigrams and bigrams, capped at 8,000 features,
`sublinear_tf=True` so word count doesn't dominate over word rarity.
Bigrams matter here: "act now" and "verify account" carry more signal as
phrases than as separate words.

## Model comparison

Three models trained on the same TF-IDF features, 80/20 train/test split,
stratified by label:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 0.965 | 0.955 | 0.958 | 0.956 |
| Logistic Regression | **0.977** | **0.966** | **0.977** | **0.971** |
| Random Forest | 0.966 | 0.952 | 0.963 | 0.957 |

Logistic Regression won and is the model the app loads.

Naive Bayes assumes word features are independent given the class, which
is a rough approximation for TF-IDF vectors where related words
(`verify`, `account`, `suspended`) tend to show up together in the same
phishing emails. Logistic Regression doesn't need that assumption; it
learns a weighted combination of the TF-IDF features directly, which fits
this kind of sparse, high-dimensional text data well. Random Forest builds
decision trees over the same features, but with 8,000 sparse TF-IDF
columns, individual trees don't split as cleanly as a linear model can
separate them, so it lands in about the same range as Naive Bayes rather
than ahead of Logistic Regression.

Confusion matrix for the winning model (rows = actual, columns =
predicted, order is [legitimate, phishing]):

```
[[938  22]
 [ 15 625]]
```

22 legitimate emails misclassified as phishing, 15 phishing emails missed,
out of 1,600 held-out test emails.

## Rule-based features

The ML model reads word patterns. It won't necessarily catch a phishing
email that avoids common phishing vocabulary but still uses a fake PayPal
domain with an IP-address link. So alongside the model, `src/features.py`
checks for:

- **Urgency language** — phrases like "verify your account", "will be
  suspended", "act now", "within 24 hours"
- **Generic greetings** — "Dear Customer" instead of a real name
- **Money/prize hooks** — "you have won", "claim your reward", "tax refund"
- **Suspicious URLs** — raw IP addresses instead of domain names, unusually
  deep subdomain chains, high-risk TLDs (`.ru`, `.tk`, `.xyz`, etc.), the
  `user@domain` redirect trick
- **Lookalike domains** — display name says "PayPal" but the domain doesn't
  match PayPal's real domain (e.g. `paypa1-secure.net`)
- **Sender/display name mismatch** — same idea, checked against the actual
  sending address rather than a link

Each check returns a plain-language reason string, which is what shows up
in the "what triggered this classification" section of the results.

The final verdict blends the ML probability with the rule score: if three
or more rule-based checks fire, the phishing probability gets nudged up
by a few percentage points (capped at 18), on the reasoning that a
message hitting several red flags at once deserves a small extra push
even if the wording alone looked fairly ordinary.

## Testing against real, unseen samples

The plan called for testing against PhishTank's live archive, which needs
API access this environment doesn't have. Instead, I set aside 150 real
phishing emails and 150 real legitimate emails that were never touched
during training or the original train/test split, and ran the same
false-positive/false-negative measurement against them:

- False negative rate (missed phishing): 2/150 = **1.3%**
- False positive rate (legitimate flagged as phishing): 6/150 = **4.0%**

A false positive costs someone a moment's confusion double-checking an
email that was actually fine. A false negative costs credentials or
money. The model leans toward catching more phishing at the cost of a
few extra false alarms, which is the right trade-off for this use case.

If you do have PhishTank API access, `tests/evaluate_real_samples.py`
has a note on how to point it at `verified_online.csv` instead.

## Running it

```bash
pip install -r requirements.txt
cd src && python3 train.py        # trains all three models, saves the best one
cd .. && python3 app.py           # starts the Flask app on localhost:5000
```

Open `http://localhost:5000`, paste an email, click Analyse.

There's also a JSON endpoint at `POST /api/analyse`:

```bash
curl -X POST http://localhost:5000/api/analyse \
  -H "Content-Type: application/json" \
  -d '{"subject": "Verify your account", "body": "...", "display_name": "PayPal", "sender_email": "support@paypa1-secure.net"}'
```

## Project structure

```
phishing-detector/
├── app.py                       Flask web app
├── requirements.txt
├── data/
│   └── phishing_email_dataset.csv
├── src/
│   ├── preprocess.py             text cleaning
│   ├── features.py                rule-based checks
│   ├── train.py                   trains and compares the three models
│   └── predict.py                 combined ML + rule verdict used by the app
├── models/                        saved model, vectoriser, metrics (after training)
├── templates/index.html
├── static/style.css
└── tests/
    └── evaluate_real_samples.py   false positive/negative check on held-out real emails
```

## Limitations

The dataset skews toward older, well-documented phishing campaigns
(Nazario's corpus is from the mid-2000s onward). Newer phishing styles,
especially ones written with AI assistance that avoid the classic urgency
phrasing, would rely more on the rule-based URL and domain checks than
the TF-IDF model, since the word patterns those campaigns use haven't
necessarily shown up in this training data.
