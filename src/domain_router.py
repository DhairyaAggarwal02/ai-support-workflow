ECOMMERCE_KEYWORDS = [
    "shipping", "delivery", "package", "order", "return", "refund",
    "damaged item", "customs", "express shipping", "standard shipping",
    "item", "checkout", "arrived", "lost package"
]

SAAS_KEYWORDS = [
    "login", "log in", "password", "account", "invoice", "api",
    "error code", "endpoint", "request id", "browser", "cache",
    "sync", "data syncing", "integration", "2fa", "two-factor",
    "plan", "trial", "billing dashboard", "dashboard", "subscription",
    "upgrade", "downgrade"
]


def route_domain(query: str):
    query_lower = query.lower()

    ecommerce_matches = [
        keyword for keyword in ECOMMERCE_KEYWORDS
        if keyword in query_lower
    ]

    saas_matches = [
        keyword for keyword in SAAS_KEYWORDS
        if keyword in query_lower
    ]

    ecommerce_score = len(ecommerce_matches)
    saas_score = len(saas_matches)

    if saas_score > ecommerce_score:
        return {
            "domain": "saas",
            "domain_confidence": round(saas_score / max(saas_score + ecommerce_score, 1), 2),
            "reason": f"SaaS keywords matched: {saas_matches}"
        }

    if ecommerce_score > saas_score:
        return {
            "domain": "ecommerce",
            "domain_confidence": round(ecommerce_score / max(saas_score + ecommerce_score, 1), 2),
            "reason": f"E-commerce keywords matched: {ecommerce_matches}"
        }

    return {
        "domain": "ecommerce",
        "domain_confidence": 0.5,
        "reason": "No strong domain match; defaulted to ecommerce"
    }


if __name__ == "__main__":
    test_queries = [
        "Where can I download my invoice?",
        "My API request failed with error code 401",
        "I cannot log into my account",
        "How do I reset my password?",
        "My package has not arrived",
        "Can I return a damaged item?"
    ]

    for query in test_queries:
        print("=" * 60)
        print(query)
        print(route_domain(query))