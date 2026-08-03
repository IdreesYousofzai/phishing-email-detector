"""
features.py
Rule-based signals that sit alongside the ML model. These catch patterns
that TF-IDF word frequencies alone can miss - a phishing email rewritten
with unusual vocabulary still keeps the same urls, name/domain mismatch
and urgency structure.

Every function returns a plain fact (bool / list / float) plus a short
human-readable reason string, so the Flask UI can show "why" an email
was flagged instead of just a black-box score.
"""

import re
from urllib.parse import urlparse

# --- Urgency / social-engineering language ------------------------------

URGENT_PHRASES = [
    "verify your account", "account will be suspended", "account suspended",
    "confirm your identity", "unusual activity", "urgent action required",
    "act now", "immediate action", "click here immediately",
    "your account has been limited", "security alert", "unauthorized login",
    "update your information", "failure to comply", "within 24 hours",
    "within 48 hours", "will be closed", "will be deactivated",
    "verify now", "confirm now", "restore access", "reactivate your account",
    "payment declined", "invoice overdue", "final notice", "response required",
]

GENERIC_GREETINGS = [
    "dear customer", "dear user", "dear valued customer", "dear member",
    "dear account holder", "dear sir/madam", "dear sir madam", "hello dear",
]

MONEY_HOOKS = [
    "won a prize", "you have won", "claim your reward", "free gift",
    "lottery", "tax refund", "wire transfer", "bank details",
    "credit card details", "social security number", "gift card",
]

SUSPICIOUS_TLDS = {
    ".ru", ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click",
    ".work", ".loan", ".men", ".zip", ".review",
}

KNOWN_BRANDS = [
    # brand name -> plausible legitimate domains
    ("paypal", ["paypal.com"]),
    ("amazon", ["amazon.com", "amazon.co.uk"]),
    ("apple", ["apple.com"]),
    ("microsoft", ["microsoft.com", "outlook.com", "live.com"]),
    ("netflix", ["netflix.com"]),
    ("google", ["google.com", "gmail.com"]),
    ("bank of america", ["bankofamerica.com"]),
    ("hsbc", ["hsbc.co.uk", "hsbc.com"]),
    ("dhl", ["dhl.com"]),
    ("royal mail", ["royalmail.com"]),
    ("hmrc", ["hmrc.gov.uk", "gov.uk"]),
]

URL_RE = re.compile(r"https?://[^\s\"'<>]+|www\.[^\s\"'<>]+")
IP_URL_RE = re.compile(r"^https?://\d{1,3}(\.\d{1,3}){3}")



def _contains_any(text: str, phrases) -> list:
    text_l = text.lower()
    return [p for p in phrases if p in text_l]



def check_urgency_language(text: str) -> dict:
    hits = _contains_any(text, URGENT_PHRASES)
    return {
        "triggered": len(hits) > 0,
        "count": len(hits),
        "matches": hits[:5],
        "reason": f"Urgency/pressure language found: {', '.join(hits[:3])}" if hits else None,
    }



def check_generic_greeting(text: str) -> dict:
    hits = _contains_any(text, GENERIC_GREETINGS)
    return {
        "triggered": len(hits) > 0,
        "matches": hits[:3],
        "reason": f"Generic greeting instead of a named recipient: '{hits[0]}'" if hits else None,
    }



def check_money_hooks(text: str) -> dict:
    hits = _contains_any(text, MONEY_HOOKS)
    return {
        "triggered": len(hits) > 0,
        "matches": hits[:5],
        "reason": f"Financial/prize bait language found: {', '.join(hits[:3])}" if hits else None,
    }



def extract_urls(text: str) -> list:
    return URL_RE.findall(text)



def analyse_url(url: str) -> dict:
    """Flag a single URL for common phishing patterns."""
    reasons = []
    raw = url
    if not url.startswith("http"):
        url = "http://" + url

    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
    except Exception:
        return {"url": raw, "suspicious": True, "reasons": ["Could not parse URL"]}

    # IP address instead of a domain name
    if IP_URL_RE.match(url):
        reasons.append("Uses a raw IP address instead of a domain name")

    # Excessive subdomains e.g. secure.login.paypal.verify-account.ru
    subdomain_count = host.count(".")
    if subdomain_count >= 3:
        reasons.append(f"Unusually deep subdomain chain ({host})")

    # Suspicious TLD
    for tld in SUSPICIOUS_TLDS:
        if host.endswith(tld):
            reasons.append(f"Uses an uncommon/high-risk top-level domain ({tld})")
            break

    # Lookalike brand in the hostname but wrong root domain
    # e.g. "paypa1-secure.net" contains a near-miss of "paypal"
    for brand, legit_domains in KNOWN_BRANDS:
        brand_token = brand.replace(" ", "")
        if brand_token[:5] in host.replace("-", "").replace(".", "") and not any(
            host == d or host.endswith("." + d) for d in legit_domains
        ):
            reasons.append(f"Domain resembles '{brand}' but does not match its real domain")
            break

    # @ symbol trick: http://real-looking-text@malicious.com
    if "@" in raw:
        reasons.append("Contains an '@' redirect trick in the URL")

    return {"url": raw, "suspicious": len(reasons) > 0, "reasons": reasons}


def check_urls(text: str) -> dict:
    urls = extract_urls(text)
    analysed = [analyse_url(u) for u in urls]
    suspicious = [a for a in analysed if a["suspicious"]]
    reason = None
    if suspicious:
        reason = f"{len(suspicious)} suspicious URL(s) found, e.g. {suspicious[0]['url']} ({suspicious[0]['reasons'][0]})"
    return {
        "triggered": len(suspicious) > 0,
        "total_urls": len(urls),
        "suspicious_urls": suspicious,
        "reason": reason,
    }



def check_sender_mismatch(display_name: str, sender_email: str) -> dict:
    """Does the display name claim to be a known brand while the actual
    sending domain does not match that brand's real domain?
    e.g. display name 'PayPal Support' but sender is 'support@paypa1-secure.net'
    """
    if not display_name or not sender_email or "@" not in sender_email:
        return {"triggered": False, "reason": None}

    display_l = display_name.lower()
    domain = sender_email.split("@")[-1].lower().strip()

    for brand, legit_domains in KNOWN_BRANDS:
        if brand in display_l:
            matches_legit = any(domain == d or domain.endswith("." + d) for d in legit_domains)
            if not matches_legit:
                return {
                    "triggered": True,
                    "reason": (
                        f"Display name claims to be '{brand.title()}' but the sending "
                        f"domain is '{domain}', which does not match {brand.title()}'s real domain"
                    ),
                }
    return {"triggered": False, "reason": None}



def extract_rule_features(subject: str, body: str, display_name: str = "", sender_email: str = "") -> dict:
    """Run every rule check and return a single combined result, including
    a 0-5 'rule score' used as a lightweight signal alongside the ML model."""
    full_text = f"{subject} {body}"

    urgency = check_urgency_language(full_text)
    greeting = check_generic_greeting(full_text)
    money = check_money_hooks(full_text)
    urls = check_urls(full_text)
    sender = check_sender_mismatch(display_name, sender_email)

    triggered_reasons = [
        r["reason"] for r in [urgency, greeting, money, urls, sender] if r.get("reason")
    ]

    rule_score = sum([
        urgency["triggered"],
        greeting["triggered"],
        money["triggered"],
        urls["triggered"],
        sender["triggered"],
    ])

    return {
        "urgency": urgency,
        "greeting": greeting,
        "money_hooks": money,
        "urls": urls,
        "sender_mismatch": sender,
        "rule_score": rule_score,          # 0-5
        "rule_score_max": 5,
        "triggered_reasons": triggered_reasons,
    }


if __name__ == "__main__":
    demo_body = (
        "Dear Customer, we detected unusual activity on your account. "
        "Your account will be suspended within 24 hours unless you verify your account now. "
        "Click here: http://paypa1-secure-login.ru/verify"
    )
    result = extract_rule_features(
        subject="URGENT: Account Verification Needed",
        body=demo_body,
        display_name="PayPal Support",
        sender_email="support@paypa1-secure-login.ru",
    )
    import json
    print(json.dumps(result, indent=2))
