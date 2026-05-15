from src.workflow import SupportWorkflow
from src.logger import log_result


def main():
    workflow = SupportWorkflow()

    print("\nAI Support Workflow Demo")
    print("Type a customer question.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Customer query: ")

        if query.lower().strip() in ["exit", "quit"]:
            print("Exiting demo.")
            break

        result = workflow.run(query)
        log_result(result)

        print("\n" + "=" * 80)
        print("Intent:", result["intent"])
        print("Intent Confidence:", result["intent_confidence"])
        print("Risk Level:", result["risk_level"])
        print("Risk Reason:", result["risk_reason"])
        print("Retrieval Score:", result["retrieval_score"])
        print("Retrieved Source:", result["retrieved_source"])
        print("Action:", result["action"])
        print("ROI:", result["roi"])
        print("\nResponse:")
        print(result["response"])
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()