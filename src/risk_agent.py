HIGH_RISK_KEYWORDS = [
    "lawsuit", "legal", "lawyer", "sue", "chargeback",
    "fraud", "scam", "angry", "furious", "complaint",
    "medication", "pregnant", "side effect", "allergic",
    "hospital", "doctor", "medical", "unsafe",
    "duplicate charge", "charged twice"
]


MEDIUM_RISK_KEYWORDS = [
    "refund", "return", "cancel", "damaged", "missing",
    "late", "lost", "subscription", "billing"
]


def assess_risk(query: str, retrieval_score: float = 0.0):
    query_lower = query.lower()

    high_matches = [
        keyword for keyword in HIGH_RISK_KEYWORDS
        if keyword in query_lower
    ]

    medium_matches = [
        keyword for keyword in MEDIUM_RISK_KEYWORDS
        if keyword in query_lower
    ]

    if high_matches:
        return {
            "risk_level": "high",
            "should_escalate": True,
            "reason": f"High-risk keywords detected: {high_matches}"
        }

    if retrieval_score < 0.15:
        return {
            "risk_level": "medium",
            "should_escalate": True,
            "reason": "Low retrieval confidence"
        }

    if medium_matches:
        return {
            "risk_level": "medium",
            "should_escalate": False,
            "reason": f"Medium-risk keywords detected: {medium_matches}"
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