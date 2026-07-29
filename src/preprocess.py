"""
preprocess.py
Text cleaning pipeline for the phishing detector.

Takes raw email text (subject + body) and reduces it to a clean, lowercase,
stopword-free string of tokens ready for TF-IDF vectorisation.
"""

import re

# A compact stop word list. Not using an external NLTK download since the
# training/inference environment may not have internet access to fetch
# NLTK corpora - keeping this dependency-free.
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "did", "do", "does", "doing", "don't", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves",
}

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML tags but keep the visible text between them."""
    return HTML_TAG_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    """Remove raw URLs from the body before TF-IDF (URLs are handled
    separately by the rule-based feature extractor)."""
    return URL_RE.sub(" ", text)


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline:
    1. lowercase
    2. strip HTML tags
    3. strip URLs
    4. remove punctuation / digits (keep letters and spaces only)
    5. tokenise on whitespace
    6. drop stop words and single-character tokens
    7. rejoin into a single string
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = strip_html(text)
    text = strip_urls(text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = text.split(" ")
    tokens = [t for t in tokens if t and t not in STOP_WORDS and len(t) > 1]

    return " ".join(tokens)


def combine_subject_body(subject: str, body: str) -> str:
    """Emails are classified on subject + body together - the subject line
    often carries strong urgency signals ('Account Suspended!!!')."""
    subject = subject or ""
    body = body or ""
    return f"{subject} {subject} {body}"  # subject weighted x2 (short but signal-dense)


if __name__ == "__main__":
    sample = "Dear Customer, <b>Your account will be SUSPENDED!!</b> Visit http://paypa1-secure.net now."
    print(clean_text(sample))
