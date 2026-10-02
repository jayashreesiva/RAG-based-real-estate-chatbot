import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"


def _value(row, column, default="Not specified"):
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return str(value).strip()


def _row_document(source_type, row):
    labels = {
        "property": [
            ("Property ID", "property_id"), ("Type", "property_type"), ("BHK", "bhk"),
            ("Price", "price_lakhs", " lakhs"), ("Area", "area_sqft", " sq.ft"),
            ("Location", "location"), ("Street", "street"), ("City", "city"),
            ("Pincode", "pincode"), ("Bathrooms", "bathrooms"), ("Property Age", "property_age", " years"),
            ("Builder", "builder"), ("Description", "description"), ("Broker ID", "broker_id"),
        ],
        "land": [
            ("Land ID", "land_id"), ("Type", "land_type"), ("Location", "location"),
            ("Street", "street"), ("City", "city"), ("Pincode", "pincode"),
            ("Area", "area_sqft", " sq.ft"), ("Price", "price_lakhs", " lakhs"),
            ("Price per sq.ft", "price_per_sqft"), ("Facing", "facing"),
            ("Description", "description"), ("Broker ID", "broker_id"),
        ],
        "school": [("School ID", "school_id"), ("Name", "school_name"), ("Area", "area"), ("Street", "street"), ("Address", "address"), ("City", "city"), ("Pincode", "pincode"), ("Type", "school_type"), ("Description", "description")],
        "college": [("College ID", "college_id"), ("Name", "college_name"), ("Area", "area"), ("Street", "street"), ("Address", "address"), ("City", "city"), ("Pincode", "pincode"), ("Type", "college_type"), ("Description", "description")],
        "hospital": [("Hospital ID", "hospital_id"), ("Name", "hospital_name"), ("Area", "area"), ("Street", "street"), ("Address", "address"), ("City", "city"), ("Pincode", "pincode"), ("Type", "hospital_type"), ("Description", "description")],
        "supermarket": [("Supermarket ID", "supermarket_id"), ("Name", "supermarket_name"), ("Area", "area"), ("Street", "street"), ("Address", "address"), ("City", "city"), ("Pincode", "pincode"), ("Description", "description")],
        "transport": [("Transport ID", "transport_id"), ("Name", "transport_name"), ("Type", "transport_type"), ("Area", "area"), ("Street", "street"), ("Address", "address"), ("City", "city"), ("Pincode", "pincode"), ("Description", "description")],
        "location": [("Location ID", "location_id"), ("Area", "area"), ("City", "city"), ("Pincode", "pincode"), ("Latitude", "latitude"), ("Longitude", "longitude"), ("Nearby Areas", "nearby_areas"), ("Description", "description")],
        "broker": [("Broker ID", "broker_id"), ("Broker Name", "broker_name"), ("Agency", "agency_name"), ("Phone", "phone"), ("Email", "email"), ("Areas Served", "areas_served"), ("Property Types", "property_types")],
    }
    source_id_column = {
        "property": "property_id", "land": "land_id", "school": "school_id", "college": "college_id",
        "hospital": "hospital_id", "supermarket": "supermarket_id", "transport": "transport_id",
        "location": "location_id", "broker": "broker_id",
    }[source_type]
    lines = [f"Source Type: {source_type}", f"Source ID: {_value(row, source_id_column)}"]
    for label, column, *suffix in labels[source_type]:
        lines.append(f"{label}: {_value(row, column)}{suffix[0] if suffix else ''}")
    return "\n".join(lines)


def _new_dataset_documents():
    dataset_specs = {
        "properties.csv": "property", "land.csv": "land", "schools.csv": "school",
        "colleges.csv": "college", "hospitals.csv": "hospital", "supermarkets.csv": "supermarket",
        "transport.csv": "transport", "locations.csv": "location", "brokers.csv": "broker",
    }
    documents = []
    metadata = []
    for filename, source_type in dataset_specs.items():
        frame = pd.read_csv(DATA_DIR / filename)
        for _, row in frame.iterrows():
            row_data = row.to_dict()
            source_id = _value(row, next(column for column in row.index if column.endswith("_id")))
            metadata.append({"source_type": source_type, "source_id": source_id, **row_data})
            documents.append(_row_document(source_type, row))
    return documents, metadata


def _legacy_property_documents():
    documents_path = DATA_DIR / "property_documents.txt"
    dataset_path = DATA_DIR / "cleaned_dataset.csv"
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents file not found at {documents_path}! Run create_documents.py first.")
    raw_text = documents_path.read_text(encoding="utf-8")
    documents = [doc.strip() for doc in raw_text.split("\n\n-----------------------------\n\n") if doc.strip()]
    metadata = []
    if dataset_path.exists():
        frame = pd.read_csv(dataset_path)
        for index, row in frame.iterrows():
            metadata.append({
                "source_type": "property", "source_id": f"LEGACY-PROP-{index + 1:04d}",
                "property_id": index + 1, "price": float(row["price"]), "area": int(row["area"]),
                "bhk": int(row["bhk"]), "status": str(row["status"]), "bathroom": float(row["bathroom"]),
                "age": float(row["age"]), "location": str(row["location"]), "builder": str(row["builder"]),
            })
    if len(metadata) != len(documents):
        metadata = [{"source_type": "property", "source_id": f"LEGACY-PROP-{index + 1:04d}", "property_id": index + 1} for index in range(len(documents))]
    return documents, metadata


def create_faiss_vector_db():
    """
    Creates a persistent FAISS vector index using Sentence Transformers ('all-MiniLM-L6-v2').
    Eliminates all ChromaDB dependencies.
    Stores:
      - faiss_index/index.faiss  (Vector index with normalized L2 Cosine similarity)
      - faiss_index/metadata.pkl (Document texts, metadata attributes, model info)
    """
    index_path = OUTPUT_DIR / "index.faiss"
    metadata_path = OUTPUT_DIR / "metadata.pkl"

    legacy_documents, legacy_metadata = _legacy_property_documents()
    new_documents, new_metadata = _new_dataset_documents()
    documents = legacy_documents + new_documents
    metadata_list = legacy_metadata + new_metadata
    print(f"Loaded {len(legacy_documents)} legacy property documents and {len(new_documents)} new dataset documents.")

    # Load embedding model
    print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    print("Generating embeddings for all property documents...")
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    print(f"Embedding dimensions: {dimension} | Total vectors: {embeddings.shape[0]}")

    # Normalize vectors for Cosine Similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS IndexFlatIP (Inner Product on normalized vectors = Cosine Similarity)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"Total vectors stored in FAISS index: {index.ntotal}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved to: {index_path}")

    # Save metadata and document mapping
    metadata_payload = {
        "documents": documents,
        "metadata": metadata_list,
        "model_name": MODEL_NAME,
        "dimension": dimension,
        "total_documents": len(documents),
        "total_properties": sum(1 for item in metadata_list if item.get("source_type") == "property"),
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata_payload, f)
    print(f"Metadata and documents saved to: {metadata_path}")

    # Quick verification test
    test_query = "3 BHK in Anna Nagar"
    test_emb = model.encode([test_query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(test_emb)
    scores, indices = index.search(test_emb, k=1)

    print("\n--- FAISS Index Verification Test ---")
    print(f"Test Query: '{test_query}'")
    top_index = int(indices[0][0])
    top_meta = metadata_list[top_index]
    print(f"Top Match: {top_meta.get('source_type')} {top_meta.get('source_id')} | Cosine Similarity: {scores[0][0]:.4f}")
    print("Vector database created and verified successfully!")

    return index, metadata_payload

if __name__ == "__main__":
    create_faiss_vector_db()