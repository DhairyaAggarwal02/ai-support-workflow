import csv
from pathlib import Path
from datetime import datetime


LOG_PATH = Path("data/processed/query_logs.csv")


def log_result(result):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": result["query"],
        "domain": result["domain"],
        "intent": result["intent"],
        "intent_confidence": result["intent_confidence"],
        "risk_level": result["risk_level"],
        "retrieval_score": result["retrieval_score"],
        "retrieved_source": result["retrieved_source"],
        "action": result["action"],
        "estimated_minutes_saved": result["roi"]["estimated_minutes_saved"],
        "estimated_labor_cost_saved": result["roi"]["estimated_labor_cost_saved"],
        "estimated_ai_cost": result["roi"]["estimated_ai_cost"],
        "estimated_net_savings": result["roi"]["estimated_net_savings"],
    }

    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)