import pandas as pd
from pathlib import Path

from src.domain_router import route_domain
from src.retriever import KnowledgeBaseRetriever
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever


EVAL_PATH = Path("eval/support_eval_set.csv")
SUMMARY_PATH = Path("reports/retriever_comparison.csv")
DETAILS_PATH = Path("reports/retriever_comparison_detailed.csv")


def build_retrievers(retriever_class):
    return {
        "ecommerce": retriever_class(domain="ecommerce"),
        "saas": retriever_class(domain="saas"),
    }


def evaluate_retriever(name, retrievers, df):
    results = []

    for _, row in df.iterrows():
        query = row["query"]
        expected_source = row["expected_source"]

        domain_result = route_domain(query)
        predicted_domain = domain_result["domain"]

        retriever = retrievers[predicted_domain]
        retrieved = retriever.retrieve(query, top_k=3)

        top1_source = retrieved[0]["source_file"]
        top3_sources = [r["source_file"] for r in retrieved]

        results.append({
            "retriever": name,
            "query": query,
            "predicted_domain": predicted_domain,
            "domain_reason": domain_result["reason"],
            "expected_source": expected_source,
            "top1_source": top1_source,
            "top3_sources": top3_sources,
            "top1_correct": top1_source == expected_source,
            "top3_correct": expected_source in top3_sources
        })

    results_df = pd.DataFrame(results)

    return {
        "retriever": name,
        "top1_accuracy": round(results_df["top1_correct"].mean(), 4),
        "top3_accuracy": round(results_df["top3_correct"].mean(), 4)
    }, results_df


def main():
    df = pd.read_csv(EVAL_PATH)

    retriever_sets = [
        ("TF-IDF", build_retrievers(KnowledgeBaseRetriever)),
        ("BM25", build_retrievers(BM25Retriever)),
        ("Hybrid", build_retrievers(HybridRetriever)),
    ]

    summary_results = []
    detailed_results = []

    for name, retrievers in retriever_sets:
        print(f"Evaluating {name}...")
        summary, details = evaluate_retriever(name, retrievers, df)
        summary_results.append(summary)
        detailed_results.append(details)

    summary_df = pd.DataFrame(summary_results)
    detailed_df = pd.concat(detailed_results)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    detailed_df.to_csv(DETAILS_PATH, index=False)

    print("\nMulti-Domain Retriever Comparison")
    print("=" * 50)
    print(summary_df)

    print("\nSaved:")
    print(SUMMARY_PATH)
    print(DETAILS_PATH)


if __name__ == "__main__":
    main()