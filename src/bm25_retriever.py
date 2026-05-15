import re
from rank_bm25 import BM25Okapi

from src.chunker import chunk_documents


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, domain="ecommerce"):
        self.domain = domain
        self.chunks = chunk_documents(domain)
        self.texts = [chunk["text"] for chunk in self.chunks]
        self.tokenized_texts = [tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_texts)

    def retrieve(self, query, top_k=3):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        max_score = max(scores) if len(scores) > 0 else 1

        for index in ranked_indices:
            chunk = self.chunks[index]
            raw_score = float(scores[index])
            normalized_score = raw_score / max_score if max_score > 0 else 0

            results.append({
                "domain": chunk["domain"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": round(normalized_score, 3),
                "raw_score": round(raw_score, 3),
                "retriever": "bm25"
            })

        return results


if __name__ == "__main__":
    retriever = BM25Retriever(domain="ecommerce")

    test_queries = [
        "My package has not arrived after 12 business days",
        "I was charged twice",
        "Can I cancel my subscription?"
    ]

    for query in test_queries:
        print("=" * 80)
        print("Query:", query)

        results = retriever.retrieve(query)

        for result in results:
            print("-" * 40)
            print("Source:", result["source_file"])
            print("Score:", result["score"])
            print("Text:", result["text"])