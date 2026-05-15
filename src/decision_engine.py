def make_decision(intent_confidence, retrieval_score, risk_result):
    """
    Decision rules:
    - High-risk always escalates.
    - Medium-risk needs stronger retrieval confidence.
    - Very weak retrieval escalates.
    - Low intent confidence only matters when retrieval is also weak.
    """

    risk_level = risk_result["risk_level"]

    if risk_level == "high":
        return {
            "action": "escalate_to_human",
            "reason": "High-risk query (forced escalation)"
        }

    if risk_level == "medium" and retrieval_score < 0.25:
        return {
            "action": "escalate_to_human",
            "reason": f"Medium risk with weak retrieval ({retrieval_score})"
        }

    if retrieval_score < 0.05:
        return {
            "action": "escalate_to_human",
            "reason": f"Very low retrieval confidence: {retrieval_score}"
        }

    if intent_confidence < 0.35 and retrieval_score < 0.15:
        return {
            "action": "escalate_to_human",
            "reason": (
                f"Low intent confidence ({intent_confidence}) "
                f"and weak retrieval confidence ({retrieval_score})"
            )
        }

    return {
        "action": "auto_respond",
        "reason": "Confidence and risk checks passed"
    }