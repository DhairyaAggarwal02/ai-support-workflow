from pathlib import Path


KNOWLEDGE_BASE_DIR = Path("knowledge_base")


def load_markdown_files(domain: str = "ecommerce"):
    domain_path = KNOWLEDGE_BASE_DIR / domain

    if not domain_path.exists():
        raise FileNotFoundError(f"Domain folder not found: {domain_path}")

    documents = []

    for file_path in domain_path.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "domain": domain,
            "source_file": file_path.name,
            "text": text
        })

    return documents


if __name__ == "__main__":
    docs = load_markdown_files("ecommerce")

    print(f"Loaded {len(docs)} documents\n")

    for doc in docs:
        print("=" * 50)
        print("Domain:", doc["domain"])
        print("Source:", doc["source_file"])
        print("Preview:")
        print(doc["text"][:300])
        print()