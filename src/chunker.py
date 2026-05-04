from src.kb_loader import load_markdown_files

import re

def chunk_text(text, chunk_size=500, overlap=100):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())

            # start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_documents(domain="ecommerce"):
    documents = load_markdown_files(domain)
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "domain": doc["domain"],
                "source_file": doc["source_file"],
                "chunk_id": f"{doc['source_file']}_{i}",
                "text": chunk
            })

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_documents("ecommerce")

    print(f"Created {len(chunks)} chunks\n")

    for chunk in chunks[:3]:
        print("=" * 50)
        print("Chunk ID:", chunk["chunk_id"])
        print("Source:", chunk["source_file"])
        print("Text:")
        print(chunk["text"])
        print()