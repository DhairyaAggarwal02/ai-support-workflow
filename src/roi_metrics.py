def estimate_roi(action: str):
    """
    Simple ROI estimator for one support query.
    Assumptions:
    - Manual support response takes 6 minutes on average
    - Human support labor cost is $25/hour
    - AI response cost is estimated at $0.02 per query
    """

    manual_minutes_per_query = 6
    hourly_labor_cost = 25
    ai_cost_per_query = 0.02

    manual_cost_per_query = (manual_minutes_per_query / 60) * hourly_labor_cost

    if action == "auto_respond":
        minutes_saved = manual_minutes_per_query
        labor_cost_saved = manual_cost_per_query
        net_savings = labor_cost_saved - ai_cost_per_query
    else:
        minutes_saved = 0
        labor_cost_saved = 0
        net_savings = -ai_cost_per_query

    return {
        "estimated_minutes_saved": round(minutes_saved, 2),
        "estimated_labor_cost_saved": round(labor_cost_saved, 2),
        "estimated_ai_cost": round(ai_cost_per_query, 2),
        "estimated_net_savings": round(net_savings, 2)
    }