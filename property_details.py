from pathlib import Path
from math import asin, cos, radians, sin, sqrt

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROPERTY_IMAGES_DIR = PROJECT_ROOT / "property_images"
PLACEHOLDER_IMAGE = PROPERTY_IMAGES_DIR / "placeholder.svg"

FACILITY_FILES = {
    "school": "schools.csv",
    "hospital": "hospitals.csv",
    "college": "colleges.csv",
    "supermarket": "supermarkets.csv",
    "transport": "transport.csv",
}


def load_property_catalog():
    return pd.read_csv(DATA_DIR / "properties.csv")


def load_land_catalog():
    return pd.read_csv(DATA_DIR / "land.csv")


def get_property_record(result, catalog=None):
    """Return a display record tied to the retrieved result's source_id."""
    if catalog is None:
        catalog = load_property_catalog()
    metadata = dict(result.get("metadata", {}))
    source_id = str(result.get("source_id", metadata.get("source_id", "")))
    matches = catalog[catalog["property_id"].astype(str) == source_id]
    if not matches.empty:
        record = matches.iloc[0].to_dict()
    else:
        land_catalog = load_land_catalog() if source_id.startswith("LAND") else pd.DataFrame()
        land_matches = land_catalog[land_catalog["land_id"].astype(str) == source_id] if not land_catalog.empty else pd.DataFrame()
        record = land_matches.iloc[0].to_dict() if not land_matches.empty else metadata
        record.setdefault("property_id", source_id or result.get("property_id"))
        record.setdefault("property_type", record.get("land_type", "Property"))
        record.setdefault("price_lakhs", record.get("price"))
        record.setdefault("area_sqft", record.get("area"))
        record.setdefault("location", "Chennai")
        record.setdefault("description", "Retrieved property record.")
    record["source_id"] = source_id or str(record.get("property_id"))
    record["similarity"] = result.get("similarity", 0.0)
    if pd.isna(record.get("latitude")) or pd.isna(record.get("longitude")):
        locations = pd.read_csv(DATA_DIR / "locations.csv")
        location_matches = locations[locations["area"].astype(str).str.lower() == str(record.get("location", "")).lower()]
        if not location_matches.empty:
            location_row = location_matches.iloc[0]
            record["latitude"] = location_row["latitude"]
            record["longitude"] = location_row["longitude"]
    return record


def _image_candidates(record):
    property_id = str(record.get("property_id", ""))
    folders = [PROPERTY_IMAGES_DIR / property_id]
    image_folder = record.get("image_folder")
    if image_folder and not pd.isna(image_folder):
        image_path = Path(str(image_folder))
        folders.append(image_path if image_path.is_absolute() else PROJECT_ROOT / image_path)
    seen_paths = set()
    for folder in folders:
        if folder.is_dir():
            for name in ("front", "living_room", "bedroom", "kitchen", "bathroom"):
                for extension in ("jpg", "jpeg", "png", "webp"):
                    candidate = folder / f"{name}.{extension}"
                    candidate_key = str(candidate.resolve()).lower()
                    if candidate.is_file() and candidate_key not in seen_paths:
                        seen_paths.add(candidate_key)
                        yield name.replace("_", " ").title(), candidate, False


def _demo_image_fallback(record):
    demo_images = []
    for folder in sorted(PROPERTY_IMAGES_DIR.glob("PROP*")):
        candidate = folder / "front.jpg"
        if candidate.is_file():
            demo_images.append(candidate)
    if not demo_images:
        return None
    property_key = str(record.get("property_id", record.get("source_id", "")))
    digits = "".join(character for character in property_key if character.isdigit())
    image_index = int(digits or 0) % len(demo_images)
    return demo_images[image_index]


def get_property_images(record):
    images = list(_image_candidates(record))
    if images:
        return images
    property_key = str(record.get("source_id", record.get("property_id", ""))).upper()
    if property_key.startswith("LEGACY-PROP"):
        demo_image = _demo_image_fallback(record)
        if demo_image:
            return [("", demo_image, False)]
    return [("", PLACEHOLDER_IMAGE, True)]


def get_search_areas(location):
    locations = pd.read_csv(DATA_DIR / "locations.csv")
    matches = locations[locations["area"].astype(str).str.lower() == str(location).lower()]
    if matches.empty:
        return [location]
    nearby = [value.strip() for value in str(matches.iloc[0]["nearby_areas"]).split(";") if value.strip()]
    return [str(matches.iloc[0]["area"])] + [value for value in nearby if value.lower() != str(location).lower()]


def calculate_distance(lat1, lon1, lat2, lon2):
    """Return approximate great-circle distance in kilometers."""
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    haversine = sin(delta_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(delta_lon / 2) ** 2
    return 2 * earth_radius_km * asin(sqrt(haversine))


def get_nearby_facilities(location):
    areas = {area.lower() for area in get_search_areas(location)}
    facilities = {}
    for facility_type, filename in FACILITY_FILES.items():
        frame = pd.read_csv(DATA_DIR / filename)
        facilities[facility_type] = frame[frame["area"].astype(str).str.lower().isin(areas)].to_dict("records")
    return facilities


def get_nearby_facilities_for_property(property_record, limit=3):
    """Load and rank facilities only for the selected property's coordinates."""
    property_latitude = property_record.get("latitude")
    property_longitude = property_record.get("longitude")
    if pd.isna(property_latitude) or pd.isna(property_longitude):
        return {facility_type: [] for facility_type in FACILITY_FILES}

    facilities = {}
    for facility_type, filename in FACILITY_FILES.items():
        frame = pd.read_csv(DATA_DIR / filename)
        frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
        frame = frame.dropna(subset=["latitude", "longitude"]).copy()
        frame["distance_km"] = frame.apply(
            lambda row: calculate_distance(property_latitude, property_longitude, row["latitude"], row["longitude"]),
            axis=1,
        )
        facilities[facility_type] = frame.sort_values("distance_km").head(limit).to_dict("records")
    return facilities


def get_broker(broker_id):
    if not broker_id or pd.isna(broker_id):
        return None
    brokers = pd.read_csv(DATA_DIR / "brokers.csv")
    matches = brokers[brokers["broker_id"].astype(str) == str(broker_id)]
    return matches.iloc[0].to_dict() if not matches.empty else None