import pandas as pd
from pathlib import Path

from src.workflow import SupportWorkflow


EVAL_PATH = Path("eval/support_eval_set.csv")
REPORT_PATH = Path("reports/workflow_eval_report.csv")
SUMMARY_PATH = Path("reports/workflow_eval_summary.csv")


def main():
    df = pd.read_csv(EVAL_PATH)

    workflow = SupportWorkflow()

    results = []

    for _, row in df.iterrows():
        result = workflow.run(row["query"])

        predicted_action = result["action"]
        predicted_source = result["retrieved_source"].split("/")[-1]

        expected_action = row["expected_action"]
        expected_source = row["expected_source"]

        action_correct = predicted_action == expected_action
        source_correct = predicted_source == expected_source

        false_automation = (
            expected_action == "escalate_to_human"
            and predicted_action == "auto_respond"
        )

        unnecessary_escalation = (
            expected_action == "auto_respond"
            and predicted_action == "escalate_to_human"
        )

        results.append({
            "query": row["query"],
            "expected_action": expected_action,
            "predicted_action": predicted_action,
            "action_correct": action_correct,
            "expected_source": expected_source,
            "predicted_source": predicted_source,
            "source_correct": source_correct,
            "false_automation": false_automation,
            "unnecessary_escalation": unnecessary_escalation,
            "domain": result["domain"],
            "intent": result["intent"],
            "intent_confidence": result["intent_confidence"],
            "risk_level": result["risk_level"],
            "risk_reason": result["risk_reason"],
            "decision_reason": result.get("decision_reason", ""),
            "retrieval_score": result["retrieval_score"],
        })

    results_df = pd.DataFrame(results)

    total_cases = len(results_df)
    action_accuracy = results_df["action_correct"].mean()
    source_accuracy = results_df["source_correct"].mean()
    false_automation_rate = results_df["false_automation"].mean()
    unnecessary_escalation_rate = results_df["unnecessary_escalation"].mean()

    total_expected_escalations = (results_df["expected_action"] == "escalate_to_human").sum()
    total_expected_auto = (results_df["expected_action"] == "auto_respond").sum()

    false_automation_count = results_df["false_automation"].sum()
    unnecessary_escalation_count = results_df["unnecessary_escalation"].sum()

    false_automation_among_escalations = (
        false_automation_count / total_expected_escalations
        if total_expected_escalations else 0
    )

    unnecessary_escalation_among_auto = (
        unnecessary_escalation_count / total_expected_auto
        if total_expected_auto else 0
    )

    summary = {
        "total_cases": total_cases,
        "action_accuracy": round(action_accuracy, 4),
        "source_accuracy": round(source_accuracy, 4),
        "false_automation_count": int(false_automation_count),
        "false_automation_rate_all_cases": round(false_automation_rate, 4),
        "false_automation_rate_expected_escalations": round(false_automation_among_escalations, 4),
        "unnecessary_escalation_count": int(unnecessary_escalation_count),
        "unnecessary_escalation_rate_all_cases": round(unnecessary_escalation_rate, 4),
        "unnecessary_escalation_rate_expected_auto": round(unnecessary_escalation_among_auto, 4),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(REPORT_PATH, index=False)
    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False)

    print("\nWorkflow Evaluation")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nSaved detailed report to {REPORT_PATH}")
    print(f"Saved summary report to {SUMMARY_PATH}")

    print("\nFalse Automations:")
    false_automations = results_df[results_df["false_automation"] == True]

    if false_automations.empty:
        print("No false automations found.")
    else:
        print(false_automations[[
            "query",
            "expected_action",
            "predicted_action",
            "risk_level",
            "risk_reason",
            "decision_reason",
            "retrieval_score"
        ]])


if __name__ == "__main__":
    main()