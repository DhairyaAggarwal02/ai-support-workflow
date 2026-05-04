from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunker import chunk_documents


class KnowledgeBaseRetriever:
    def __init__(self, domain="ecommerce"):
        self.domain = domain
        self.chunks = chunk_documents(domain)
        self.texts = [chunk["text"] for chunk in self.chunks]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=3):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix)[0]

        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            chunk = self.chunks[index]

            results.append({
                "domain": chunk["domain"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": round(float(similarities[index]), 3)
            })

        return results


if __name__ == "__main__":
    retriever = KnowledgeBaseRetriever(domain="ecommerce")

    test_query = "My package has not arrived after 12 business days"
    results = retriever.retrieve(test_query)

    print(f"Query: {test_query}\n")

    for result in results:
        print("=" * 60)
        print("Source:", result["source_file"])
        print("Score:", result["score"])
        print("Text:")
        print(result["text"])