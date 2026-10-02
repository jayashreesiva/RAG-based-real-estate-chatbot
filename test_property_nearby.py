from pathlib import Path

from property_details import (
    calculate_distance,
    get_nearby_facilities_for_property,
    get_property_images,
    get_property_record,
    load_property_catalog,
)
from rag_chatbot import RealEstateRAG


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    catalog = load_property_catalog()
    rag = RealEstateRAG(index_dir=PROJECT_ROOT / "faiss_index", require_gemini=False)
    results, _ = rag.retrieve("PROP001 apartment in Velachery", top_k=25, min_similarity=0.0)
    selected_result = next(item for item in results if item.get("source_id") == "PROP001")
    selected_property = get_property_record(selected_result, catalog)
    assert selected_property["property_id"] == "PROP001"
    print("PASS: selected property_id")

    images = get_property_images(selected_property)
    assert images and all(path.exists() for _, path, _ in images)
    print("PASS: selected property images")

    facilities = get_nearby_facilities_for_property(selected_property)
    for facility_type in ["school", "college", "hospital", "supermarket", "transport"]:
        rows = facilities[facility_type]
        assert len(rows) <= 3
        assert rows == sorted(rows, key=lambda row: row["distance_km"])
        assert all(row["distance_km"] >= 0 for row in rows)
        print(f"PASS: {facility_type} distances sorted and limited")

    assert calculate_distance(12.0, 80.0, 12.0, 80.0) == 0
    unselected_property = get_property_record({"source_id": "PROP002", "metadata": {}}, catalog)
    assert unselected_property["property_id"] == "PROP002"
    assert "nearby_facilities" not in unselected_property
    print("PASS: unselected property does not trigger nearby lookup")

    land_results, _ = rag.retrieve("LAND001 land in Velachery", top_k=25, min_similarity=0.0)
    land_result = next(item for item in land_results if item.get("source_id") == "LAND001")
    land_record = get_property_record(land_result, catalog)
    land_facilities = get_nearby_facilities_for_property(land_record)
    assert land_record["property_id"] == "LAND001"
    assert land_record["land_type"] == "Residential Plot"
    assert all(len(rows) <= 3 for rows in land_facilities.values())
    print("PASS: selected land property nearby lookup")
    print("ALL PROPERTY NEARBY TESTS PASSED")


if __name__ == "__main__":
    main()