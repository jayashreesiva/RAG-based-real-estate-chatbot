from pathlib import Path

from property_details import (
    get_broker,
    get_nearby_facilities,
    get_property_images,
    get_property_record,
    load_property_catalog,
)
from rag_chatbot import RealEstateRAG


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    catalog = load_property_catalog()
    required_columns = {
        "property_id", "property_type", "bhk", "price_lakhs", "area_sqft", "location", "street",
        "city", "pincode", "bathrooms", "property_age", "builder", "description", "broker_id", "image_folder",
    }
    assert required_columns.issubset(catalog.columns)

    rag = RealEstateRAG(index_dir=PROJECT_ROOT / "faiss_index", require_gemini=False)
    results, _ = rag.retrieve("PROP001 2 BHK apartment in Velachery", top_k=25, min_similarity=0.0)
    property_result = next((item for item in results if item.get("source_id") == "PROP001"), None)
    assert property_result is not None, "PROP001 was not retrieved from FAISS"
    record = get_property_record(property_result, catalog)
    assert record["property_id"] == "PROP001"
    print("PASS: property retrieval and property_id")

    images = get_property_images(record)
    assert images and all(path.exists() for _, path, _ in images)
    print("PASS: image_folder and image fallback")

    brokers = {str(value) for value in catalog["broker_id"]}
    broker = get_broker(record["broker_id"])
    assert record["broker_id"] in brokers and broker is not None
    print("PASS: broker_id relationship")

    facilities = get_nearby_facilities(record["location"])
    assert all(facilities[facility_type] for facility_type in ["school", "hospital", "college", "supermarket", "transport"])
    print("PASS: nearby facility lookup")

    missing_image_record = {"property_id": "MISSING-DEMO", "location": "Velachery"}
    fallback_images = get_property_images(missing_image_record)
    assert fallback_images and fallback_images[0][2] is True and fallback_images[0][1].exists()
    print("PASS: missing images do not crash")
    print("ALL PROPERTY DETAILS TESTS PASSED")


if __name__ == "__main__":
    main()