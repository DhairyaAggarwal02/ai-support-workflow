from collections import defaultdict

from src.retriever import KnowledgeBaseRetriever
from src.bm25_retriever import BM25Retriever


class HybridRetriever:
    def __init__(self, domain="ecommerce", tfidf_weight=0.4, bm25_weight=0.6):
        self.domain = domain
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight

        self.tfidf_retriever = KnowledgeBaseRetriever(domain=domain)
        self.bm25_retriever = BM25Retriever(domain=domain)

    def retrieve(self, query, top_k=3):
        tfidf_results = self.tfidf_retriever.retrieve(query, top_k=10)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=10)

        combined = defaultdict(lambda: {
            "domain": None,
            "source_file": None,
            "chunk_id": None,
            "text": None,
            "tfidf_score": 0.0,
            "bm25_score": 0.0,
            "score": 0.0,
            "retriever": "hybrid"
        })

        for result in tfidf_results:
            chunk_id = result["chunk_id"]
            combined[chunk_id].update({
                "domain": result["domain"],
                "source_file": result["source_file"],
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "tfidf_score": result["score"]
            })

        for result in bm25_results:
            chunk_id = result["chunk_id"]
            combined[chunk_id].update({
                "domain": result["domain"],
                "source_file": result["source_file"],
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "bm25_score": result["score"]
            })

        final_results = []

        for item in combined.values():
            item["score"] = round(
                self.tfidf_weight * item["tfidf_score"] +
                self.bm25_weight * item["bm25_score"],
                3
            )
            final_results.append(item)

        final_results = sorted(
            final_results,
            key=lambda x: x["score"],
            reverse=True
        )

        return final_results[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever(domain="ecommerce")

    test_queries = [
        "My package has not arrived after 12 business days",
        "I was charged twice",
        "Can I cancel my subscription?",
        "How long does express delivery take?"
    ]

    for query in test_queries:
        print("=" * 80)
        print("Query:", query)

        results = retriever.retrieve(query)

        for result in results:
            print("-" * 40)
            print("Source:", result["source_file"])
            print("Hybrid Score:", result["score"])
            print("TF-IDF:", result["tfidf_score"])
            print("BM25:", result["bm25_score"])
            print("Text:", result["text"][:400])