import os
import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent

def load_faiss_store(index_dir="faiss_index"):
    """
    Loads the persistent FAISS index and associated metadata.
    """
    index_dir = Path(index_dir)
    if not index_dir.is_absolute():
        index_dir = PROJECT_ROOT / index_dir
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.pkl"

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"FAISS index files not found in '{index_dir}'! "
            f"Please run 'python create_vector_db.py' first."
        )

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        metadata_payload = pickle.load(f)

    return index, metadata_payload

def search_properties(query, top_k=5, min_similarity=0.25, model=None, index=None, metadata_payload=None):
    """
    Performs semantic vector search using FAISS and Sentence Transformers.
    Computes cosine similarity between query embedding and stored document embeddings.
    Returns a list of matching result dictionaries.
    """
    if index is None or metadata_payload is None:
        index, metadata_payload = load_faiss_store()

    if model is None:
        model = SentenceTransformer(metadata_payload.get("model_name", "all-MiniLM-L6-v2"))

    # Convert query into normalized embedding
    query_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_emb)

    # Search FAISS index
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_emb, k)

    results = []
    documents = metadata_payload["documents"]
    meta_list = metadata_payload.get("metadata", [])

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(documents):
            continue
        if score < min_similarity:
            continue

        item_meta = meta_list[idx] if idx < len(meta_list) else {}
        results.append({
            "property_id": idx + 1,
            "document": documents[idx],
            "similarity_score": float(score),
            "metadata": item_meta
        })

    return results

def main():
    print("=" * 60)
    print("FAISS Real Estate Semantic Search")
    print("=" * 60)

    try:
        index, metadata_payload = load_faiss_store()
        model = SentenceTransformer(metadata_payload.get("model_name", "all-MiniLM-L6-v2"))
        print(f"Loaded FAISS index with {index.ntotal} properties.")
    except Exception as e:
        print(f"Error loading FAISS store: {e}")
        return

    while True:
        try:
            query = input("\nEnter your real estate query (or 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query or query.lower() in ["exit", "quit"]:
            print("Exiting search. Goodbye!")
            break

        results = search_properties(query, top_k=5, model=model, index=index, metadata_payload=metadata_payload)

        if not results:
            print("No matching properties found meeting the similarity threshold.")
            continue

        print(f"\nFound {len(results)} relevant properties:\n")
        for i, res in enumerate(results, 1):
            print(f"--- Result {i} (Similarity: {res['similarity_score']:.4f} / {res['similarity_score']*100:.1f}%) ---")
            print(res["document"])
            print()

if __name__ == "__main__":
    main()