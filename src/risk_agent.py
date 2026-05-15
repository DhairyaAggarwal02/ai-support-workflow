HIGH_RISK_KEYWORDS = [
    "lawsuit", "legal", "lawyer", "sue", "chargeback",
    "fraud", "scam", "angry", "furious", "complaint",
    "medication", "pregnant", "side effect", "allergic",
    "hospital", "doctor", "medical", "unsafe",
    "duplicate charge", "charged twice"

    # financial
    "charged twice", "duplicate charge", "chargeback", "refund dispute",

    # legal
    "sue", "lawyer", "legal action", "court",

    # security
    "hacked", "unauthorized access", "account hacked",
    "fraud", "scam", "compromised", "data breach",

    # health (if applicable)
    "pregnant", "allergic reaction", "hospital"
]


MEDIUM_RISK_KEYWORDS = [
    "refund", "return", "cancel", "damaged", "missing",
    "late", "lost", "subscription", "complaint", "angry", "not satisfied", 
    "billing", "bad service"
]


def assess_risk(query: str, retrieval_score: float):
    query_lower = query.lower()

    matched_high = [
        kw for kw in HIGH_RISK_KEYWORDS
        if kw in query_lower
    ]

    matched_medium = [
        kw for kw in MEDIUM_RISK_KEYWORDS
        if kw in query_lower
    ]

    if matched_high:
        return {
            "risk_level": "high",
            "should_escalate": True,
            "reason": f"High-risk keywords detected: {matched_high}"
        }

    if matched_medium:
        return {
            "risk_level": "medium",
            "should_escalate": False,
            "reason": f"Medium-risk keywords detected: {matched_medium}"
        }

    if retrieval_score < 0.10:
        return {
            "risk_level": "medium",
            "should_escalate": True,
            "reason": "Low retrieval confidence"
        }

    return {
        "risk_level": "low",
        "should_escalate": False,
        "reason": "No major risk detected"
    }


if __name__ == "__main__":
    test_queries = [
        "Where is my package?",
        "I want to sue your company",
        "Can I take this product while pregnant?",
        "I was charged twice for my order",
        "How long does standard shipping take?"
    ]

    for query in test_queries:
        print("=" * 60)
        print("Query:", query)
        print(assess_risk(query, retrieval_score=0.8))