from src.intent_classifier import predict_intent
from src.retriever import KnowledgeBaseRetriever
from src.risk_agent import assess_risk
from src.roi_metrics import estimate_roi


class SupportWorkflow:
    def __init__(self, domain="ecommerce"):
        self.domain = domain
        self.retriever = KnowledgeBaseRetriever(domain=domain)

    def run(self, query: str):
        intent_result = predict_intent(query)

        retrieved_chunks = self.retriever.retrieve(query, top_k=3)
        top_chunk = retrieved_chunks[0]
        top_score = top_chunk["score"]

        risk_result = assess_risk(query, retrieval_score=top_score)

        if risk_result["should_escalate"]:
            action = "escalate_to_human"
            response = (
                "This request should be reviewed by a human support agent. "
                "The system found potential risk or low confidence."
            )
        else:
            action = "auto_respond"
            response = top_chunk["text"]
        
        roi_result = estimate_roi(action)

        return {
            "query": query,
            "domain": self.domain,
            "intent": intent_result["predicted_intent"],
            "intent_confidence": intent_result["confidence"],
            "risk_level": risk_result["risk_level"],
            "risk_reason": risk_result["reason"],
            "retrieval_score": top_score,
            "retrieved_source": top_chunk["source_file"],
            "action": action,
            "response": response,
            "roi": roi_result,
            "retrieved_chunks": retrieved_chunks
        }


if __name__ == "__main__":
    workflow = SupportWorkflow(domain="ecommerce")

    test_queries = [
        "How long does standard shipping take?",
        "My package has not arrived after 12 business days",
        "I was charged twice for my order",
        "Can I cancel my subscription before renewal?",
        "I want to sue your company"
    ]

    for query in test_queries:
        result = workflow.run(query)

        print("=" * 80)
        print("Query:", result["query"])
        print("Intent:", result["intent"])
        print("Intent Confidence:", result["intent_confidence"])
        print("Risk:", result["risk_level"])
        print("Risk Reason:", result["risk_reason"])
        print("Retrieval Score:", result["retrieval_score"])
        print("Source:", result["retrieved_source"])
        print("Action:", result["action"])
        print("Response:")
        print(result["response"])
        print("ROI:", result["roi"])