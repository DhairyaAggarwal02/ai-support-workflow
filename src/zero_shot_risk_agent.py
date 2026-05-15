from transformers import pipeline


RISK_LABELS = [
    "normal customer support question",
    "billing or refund dispute",
    "security or account takeover issue",
    "legal threat",
    "medical or safety concern",
    "angry or abusive customer",
    "question not covered by support policy"
]


HIGH_RISK_LABELS = {
    "billing or refund dispute",
    "security or account takeover issue",
    "legal threat",
    "medical or safety concern",
}


MEDIUM_RISK_LABELS = {
    "angry or abusive customer",
    "question not covered by support policy",
}


class ZeroShotRiskAgent:
    def __init__(self):
        self.classifier = pipeline(
            task="zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    def assess_risk(self, query: str):
        result = self.classifier(
            query,
            candidate_labels=RISK_LABELS,
            multi_label=False
        )

        top_label = result["labels"][0]
        top_score = float(result["scores"][0])

        query_lower = query.lower()

        safe_billing_patterns = [
            "download my invoice",
            "download invoice",
            "invoice",
            "receipt",
            "billing method",
            "payment method",
        ]

        safe_technical_patterns = [
            "app is not loading",
            "browser",
            "cache",
            "login page",
            "error code",
        ]

        if any(pattern in query_lower for pattern in safe_billing_patterns):
            return {
                "risk_level": "low",
                "should_escalate": False,
                "risk_label": top_label,
                "risk_score": round(top_score, 3),
                "reason": "Safe billing/support pattern detected"
            }

        if any(pattern in query_lower for pattern in safe_technical_patterns):
            return {
                "risk_level": "low",
                "should_escalate": False,
                "risk_label": top_label,
                "risk_score": round(top_score, 3),
                "reason": "Safe technical support pattern detected"
            }

        if top_label in HIGH_RISK_LABELS and top_score >= 0.75:
            return {
                "risk_level": "high",
                "should_escalate": True,
                "risk_label": top_label,
                "risk_score": round(top_score, 3),
                "reason": f"Zero-shot high-risk classification: {top_label}"
            }

        if top_label in MEDIUM_RISK_LABELS and top_score >= 0.65:
            return {
                "risk_level": "medium",
                "should_escalate": True,
                "risk_label": top_label,
                "risk_score": round(top_score, 3),
                "reason": f"Zero-shot medium-risk classification: {top_label}"
            }

        return {
            "risk_level": "low",
            "should_escalate": False,
            "risk_label": top_label,
            "risk_score": round(top_score, 3),
            "reason": "Zero-shot risk score below escalation threshold"
    }
    


if __name__ == "__main__":
    agent = ZeroShotRiskAgent()

    test_queries = [
        "How long does shipping take?",
        "Where can I download my invoice?",
        "I want to dispute this refund decision",
        "Someone else claims ownership of my account",
        "I think my account was hacked",
        "I want to sue your company",
        "Can I take this while pregnant?",
        "The app is not loading in my browser",
    ]

    for query in test_queries:
        print("=" * 80)
        print("Query:", query)
        print(agent.assess_risk(query))