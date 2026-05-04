import pandas as pd
from pathlib import Path


LOG_PATH = Path("data/processed/query_logs.csv")


def calculate_metrics():
    if not LOG_PATH.exists():
        raise FileNotFoundError("No query_logs.csv found. Run the CLI demo first.")

    df = pd.read_csv(LOG_PATH)

    total_queries = len(df)
    auto_responses = (df["action"] == "auto_respond").sum()
    escalations = (df["action"] == "escalate_to_human").sum()

    automation_rate = auto_responses / total_queries if total_queries else 0
    escalation_rate = escalations / total_queries if total_queries else 0

    total_minutes_saved = df["estimated_minutes_saved"].sum()
    total_labor_saved = df["estimated_labor_cost_saved"].sum()
    total_ai_cost = df["estimated_ai_cost"].sum()
    total_net_savings = df["estimated_net_savings"].sum()

    avg_intent_confidence = df["intent_confidence"].mean()
    avg_retrieval_score = df["retrieval_score"].mean()

    return {
        "total_queries": total_queries,
        "auto_responses": int(auto_responses),
        "escalations": int(escalations),
        "automation_rate_percent": round(automation_rate * 100, 2),
        "escalation_rate_percent": round(escalation_rate * 100, 2),
        "total_minutes_saved": round(total_minutes_saved, 2),
        "total_hours_saved": round(total_minutes_saved / 60, 2),
        "total_labor_saved": round(total_labor_saved, 2),
        "total_ai_cost": round(total_ai_cost, 2),
        "total_net_savings": round(total_net_savings, 2),
        "avg_intent_confidence": round(avg_intent_confidence, 3),
        "avg_retrieval_score": round(avg_retrieval_score, 3),
    }


if __name__ == "__main__":
    metrics = calculate_metrics()

    print("\nBusiness Metrics Dashboard")
    print("=" * 50)

    for key, value in metrics.items():
        print(f"{key}: {value}")