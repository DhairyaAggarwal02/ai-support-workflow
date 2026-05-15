from src.intent_classifier import predict_intent
from src.retriever import KnowledgeBaseRetriever
from src.risk_agent import assess_risk
from src.roi_metrics import estimate_roi
from src.domain_router import route_domain
from src.response_formatter import format_response
from src.decision_engine import make_decision
from src.zero_shot_risk_agent import ZeroShotRiskAgent

class SupportWorkflow:
    def __init__(self, domain=None):
        self.default_domain = domain
        self.retrievers = {}
        self.zero_shot_risk_agent = ZeroShotRiskAgent()
    
    def get_retriever(self, domain):
        if domain not in self.retrievers:
            self.retrievers[domain] = KnowledgeBaseRetriever(domain=domain)
        return self.retrievers[domain]

    def run(self, query: str):
        
        intent_result = predict_intent(query)

        if self.default_domain:
            domain = self.default_domain
            domain_result = {
                "domain": domain,
                "domain_confidence": 1.0,
                "reason": "Manual domain override"
            }
        else:
            domain_result = route_domain(query)
            domain = domain_result["domain"]

        retriever = self.get_retriever(domain)
        retrieved_chunks = retriever.retrieve(query, top_k=3)

        top_chunk = retrieved_chunks[0]
        top_score = top_chunk["score"]

        risk_result = self.zero_shot_risk_agent.assess_risk(query)

        intent_confidence = intent_result["confidence"]

    

        
        decision_result = make_decision(
            intent_confidence=intent_confidence,
            retrieval_score=top_score,
            risk_result=risk_result
        )

        action = decision_result["action"]

        if action == "escalate_to_human":
            response = (
                "This request should be reviewed by a human support agent. "
                f"Reason: {decision_result['reason']}"
            )
        else:
            response = format_response(
                query=query,
                chunk_text=top_chunk["text"],
                source_file=top_chunk["source_file"]
            )
        
        roi_result = estimate_roi(action)

        return {
            "query": query,
            "domain": domain,
            "domain_confidence": domain_result["domain_confidence"],
            "domain_reason": domain_result["reason"],
            "decision_reason": decision_result["reason"],
            "intent": intent_result["predicted_intent"],
            "intent_confidence": intent_result["confidence"],
            "risk_level": risk_result["risk_level"],
            "risk_reason": risk_result["reason"],
            "retrieval_score": top_score,
            "retrieved_source": f"{domain}/{top_chunk['source_file']}",
            "action": action,
            "response": response,
            "roi": roi_result,
            "retrieved_chunks": retrieved_chunks
        }


if __name__ == "__main__":
    workflow = SupportWorkflow()

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
        print("Domain:", result["domain"])
        print("Domain Confidence:", result["domain_confidence"])
        print("Domain Reason:", result["domain_reason"])
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