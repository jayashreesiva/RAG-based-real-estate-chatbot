from pathlib import Path

import pandas as pd

from generate_datasets import build_datasets


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

SCHEMAS = {
    "properties.csv": ["property_id", "property_type", "bhk", "price_lakhs", "area_sqft", "location", "street", "city", "pincode", "bathrooms", "property_age", "builder", "description", "broker_id", "image_folder", "latitude", "longitude"],
    "land.csv": ["land_id", "land_type", "area_sqft", "price_lakhs", "price_per_sqft", "location", "street", "city", "pincode", "facing", "broker_id", "description", "latitude", "longitude"],
    "schools.csv": ["school_id", "school_name", "area", "street", "address", "city", "pincode", "school_type", "description", "latitude", "longitude"],
    "colleges.csv": ["college_id", "college_name", "area", "street", "address", "city", "pincode", "college_type", "description", "latitude", "longitude"],
    "hospitals.csv": ["hospital_id", "hospital_name", "area", "street", "address", "city", "pincode", "hospital_type", "description", "latitude", "longitude"],
    "supermarkets.csv": ["supermarket_id", "supermarket_name", "area", "street", "address", "city", "pincode", "description", "latitude", "longitude"],
    "transport.csv": ["transport_id", "transport_name", "transport_type", "area", "street", "address", "city", "pincode", "description", "latitude", "longitude"],
    "locations.csv": ["location_id", "area", "city", "pincode", "latitude", "longitude", "nearby_areas", "description"],
    "brokers.csv": ["broker_id", "broker_name", "agency_name", "phone", "email", "areas_served", "property_types"],
}


def fail(message):
    raise ValueError(message)


def main():
    frames = {}
    for filename, columns in SCHEMAS.items():
        path = DATA_DIR / filename
        if not path.is_file():
            fail(f"Missing file: {path}")
        frame = pd.read_csv(path)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            fail(f"{filename} is missing columns: {missing}")
        if frame.empty:
            fail(f"{filename} has no records")
        id_column = columns[0]
        if frame[id_column].isna().any() or frame[id_column].astype(str).duplicated().any():
            fail(f"{filename} has empty or duplicate {id_column} values")
        important = [column for column in columns if column not in {"latitude", "longitude", "nearby_areas"}]
        if frame[important].isna().any().any() or (frame[important].astype(str).apply(lambda column: column.str.strip() == "").any().any()):
            fail(f"{filename} has empty important fields")
        frames[filename] = frame

    properties = frames["properties.csv"]
    land = frames["land.csv"]
    locations = frames["locations.csv"]
    brokers = frames["brokers.csv"]
    known_areas = set(locations["area"])
    known_brokers = set(brokers["broker_id"])
    for filename, frame in [("properties.csv", properties), ("land.csv", land)]:
        if not set(frame["location"]).issubset(known_areas):
            fail(f"{filename} contains an unknown location")
        if not set(frame["broker_id"]).issubset(known_brokers):
            fail(f"{filename} contains an unknown broker_id")
        if not (frame["area_sqft"] > 0).all() or not (frame["price_lakhs"] > 0).all():
            fail(f"{filename} contains invalid area or price")
    if not properties["bhk"].isin([1, 2, 3, 4]).all() or not (properties["bathrooms"] > 0).all():
        fail("properties.csv contains invalid BHK or bathroom values")
    if not (land["price_per_sqft"] > 0).all():
        fail("land.csv contains invalid price_per_sqft values")
    for filename in ["schools.csv", "colleges.csv", "hospitals.csv", "supermarkets.csv", "transport.csv"]:
        if not set(frames[filename]["area"]).issubset(known_areas):
            fail(f"{filename} contains an unknown area")
        if not frames[filename]["latitude"].between(-90, 90).all() or not frames[filename]["longitude"].between(-180, 180).all():
            fail(f"{filename} contains invalid coordinates")
    for filename in ["properties.csv", "land.csv"]:
        if not frames[filename]["latitude"].between(-90, 90).all() or not frames[filename]["longitude"].between(-180, 180).all():
            fail(f"{filename} contains invalid coordinates")
    if not locations["latitude"].between(-90, 90).all() or not locations["longitude"].between(-180, 180).all():
        fail("locations.csv contains invalid coordinates")
    if not all(locations.apply(lambda row: all(area in known_areas for area in str(row["nearby_areas"]).split("; ")), axis=1)):
        fail("locations.csv contains an unknown nearby area")
    if not all(properties["description"].str.contains("Synthetic demo")) or not all(brokers["email"].str.endswith("@example.com")):
        fail("Records are not clearly marked as demo data")
    print("DATASET VALIDATION PASSED")
    for filename, frame in frames.items():
        print(f"{filename}: {len(frame)} records")


if __name__ == "__main__":
    main()