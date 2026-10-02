from pathlib import Path

from rag_chatbot import RealEstateRAG


PROJECT_ROOT = Path(__file__).resolve().parent

TEST_CASES = [
    ("3 BHK houses in Velachery", "property"),
    ("Land in Tambaram", "land"),
    ("Schools in Velachery", "school"),
    ("Hospitals in Velachery", "hospital"),
    ("Colleges in Taramani", "college"),
    ("Supermarkets in Pallikaranai", "supermarket"),
    ("Transport facilities in Velachery", "transport"),
    ("Who is the broker for this property?", "broker"),
]


def main():
    rag = RealEstateRAG(index_dir=PROJECT_ROOT / "faiss_index", require_gemini=False)
    failures = []
    for query, expected_source_type in TEST_CASES:
        results, _ = rag.retrieve(query, top_k=3, min_similarity=0.0)
        top_result = results[0] if results else None
        source_type = top_result.get("source_type") if top_result else None
        source_id = top_result.get("source_id") if top_result else None
        similarity = top_result.get("similarity", 0.0) if top_result else 0.0
        print(f"QUERY:\n{query}\n")
        print(f"RESULT:\n{top_result['document'] if top_result else 'No result'}\n")
        print(f"SOURCE TYPE:\n{source_type}\n")
        print(f"SOURCE ID:\n{source_id}\n")
        print(f"SIMILARITY SCORE:\n{similarity:.4f}\n")
        print("-" * 60)
        if source_type != expected_source_type:
            failures.append(f"{query}: expected {expected_source_type}, got {source_type}")
    if failures:
        raise AssertionError("\n".join(failures))
    print("ALL RETRIEVAL TESTS PASSED")


if __name__ == "__main__":
    main()