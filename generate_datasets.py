from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def build_datasets():
    brokers = [
        {"broker_id": "BRK001", "broker_name": "Demo Broker 01", "agency_name": "Chennai Demo Realty", "phone": "+91-90000-00001", "email": "demo.broker01@example.com", "areas_served": "Velachery; Taramani; Adambakkam", "property_types": "Apartment; Independent House"},
        {"broker_id": "BRK002", "broker_name": "Demo Broker 02", "agency_name": "South City Demo Homes", "phone": "+91-90000-00002", "email": "demo.broker02@example.com", "areas_served": "Pallikaranai; Madipakkam; Medavakkam", "property_types": "Apartment; Villa"},
        {"broker_id": "BRK003", "broker_name": "Demo Broker 03", "agency_name": "Metro Edge Demo Properties", "phone": "+91-90000-00003", "email": "demo.broker03@example.com", "areas_served": "Anna Nagar; Perungudi; Sholinganallur", "property_types": "Apartment; Villa"},
        {"broker_id": "BRK004", "broker_name": "Demo Broker 04", "agency_name": "Tambaram Demo Estates", "phone": "+91-90000-00004", "email": "demo.broker04@example.com", "areas_served": "Tambaram; Pallikaranai; Madipakkam", "property_types": "Independent House; Villa"},
    ]

    locations = [
        {"location_id": "LOC001", "area": "Velachery", "city": "Chennai", "pincode": "600042", "latitude": 12.9792, "longitude": 80.2180, "nearby_areas": "Taramani; Adambakkam; Pallikaranai", "description": "Synthetic demo coordinates for Velachery; established residential area with access toward Taramani and Adambakkam."},
        {"location_id": "LOC002", "area": "Taramani", "city": "Chennai", "pincode": "600113", "latitude": 12.9916, "longitude": 80.2425, "nearby_areas": "Velachery; Perungudi; Adambakkam", "description": "Synthetic demo coordinates for Taramani; technology and residential corridor near Velachery and Perungudi."},
        {"location_id": "LOC003", "area": "Adambakkam", "city": "Chennai", "pincode": "600088", "latitude": 12.9827, "longitude": 80.2018, "nearby_areas": "Velachery; Madipakkam; Taramani", "description": "Synthetic demo coordinates for Adambakkam; connected residential locality near Velachery and Madipakkam."},
        {"location_id": "LOC004", "area": "Pallikaranai", "city": "Chennai", "pincode": "600100", "latitude": 12.9377, "longitude": 80.2121, "nearby_areas": "Velachery; Medavakkam; Sholinganallur", "description": "Synthetic demo coordinates for Pallikaranai; growing residential area with access to Medavakkam and Sholinganallur."},
        {"location_id": "LOC005", "area": "Madipakkam", "city": "Chennai", "pincode": "600091", "latitude": 12.9624, "longitude": 80.1986, "nearby_areas": "Adambakkam; Pallikaranai; Medavakkam", "description": "Synthetic demo coordinates for Madipakkam; residential area linked to Adambakkam, Pallikaranai and Medavakkam."},
        {"location_id": "LOC006", "area": "Anna Nagar", "city": "Chennai", "pincode": "600040", "latitude": 13.0850, "longitude": 80.2101, "nearby_areas": "Taramani; Perungudi", "description": "Synthetic demo coordinates for Anna Nagar; established central-west Chennai residential and retail area."},
        {"location_id": "LOC007", "area": "Tambaram", "city": "Chennai", "pincode": "600045", "latitude": 12.9249, "longitude": 80.1000, "nearby_areas": "Medavakkam; Pallikaranai", "description": "Synthetic demo coordinates for Tambaram; suburban residential hub with rail connectivity."},
        {"location_id": "LOC008", "area": "Perungudi", "city": "Chennai", "pincode": "600096", "latitude": 12.9600, "longitude": 80.2450, "nearby_areas": "Taramani; Sholinganallur; Velachery", "description": "Synthetic demo coordinates for Perungudi; business corridor near Taramani and Sholinganallur."},
        {"location_id": "LOC009", "area": "Sholinganallur", "city": "Chennai", "pincode": "600119", "latitude": 12.9010, "longitude": 80.2279, "nearby_areas": "Perungudi; Pallikaranai; Medavakkam", "description": "Synthetic demo coordinates for Sholinganallur; IT corridor with residential development."},
        {"location_id": "LOC010", "area": "Medavakkam", "city": "Chennai", "pincode": "600100", "latitude": 12.9172, "longitude": 80.1920, "nearby_areas": "Pallikaranai; Madipakkam; Tambaram", "description": "Synthetic demo coordinates for Medavakkam; expanding residential area connected to Pallikaranai and Tambaram."},
    ]

    properties = [
        {"property_id": "PROP001", "property_type": "Apartment", "bhk": 2, "price_lakhs": 78.0, "area_sqft": 1050, "location": "Velachery", "street": "Demo Lake View Road", "city": "Chennai", "pincode": "600042", "bathrooms": 2, "property_age": 4, "builder": "Demo Habitat Builders", "description": "Synthetic demo 2 BHK apartment in Velachery with two bathrooms, covered parking and a practical family layout.", "broker_id": "BRK001", "image_folder": "images/demo/PROP001"},
        {"property_id": "PROP002", "property_type": "Apartment", "bhk": 3, "price_lakhs": 122.0, "area_sqft": 1450, "location": "Taramani", "street": "Demo Research Park Avenue", "city": "Chennai", "pincode": "600113", "bathrooms": 3, "property_age": 2, "builder": "Demo Urban Spaces", "description": "Synthetic demo 3 BHK apartment in Taramani near the business corridor, with three bathrooms and a balcony.", "broker_id": "BRK001", "image_folder": "images/demo/PROP002"},
        {"property_id": "PROP003", "property_type": "Independent House", "bhk": 3, "price_lakhs": 135.0, "area_sqft": 1800, "location": "Adambakkam", "street": "Demo Lake Junction", "city": "Chennai", "pincode": "600088", "bathrooms": 3, "property_age": 8, "builder": "Demo Nest Homes", "description": "Synthetic demo 3 BHK independent house in Adambakkam with a private frontage, three bathrooms and room for renovation.", "broker_id": "BRK001", "image_folder": "images/demo/PROP003"},
        {"property_id": "PROP004", "property_type": "Apartment", "bhk": 2, "price_lakhs": 69.0, "area_sqft": 980, "location": "Pallikaranai", "street": "Demo Marsh View Street", "city": "Chennai", "pincode": "600100", "bathrooms": 2, "property_age": 5, "builder": "Demo South Living", "description": "Synthetic demo 2 BHK apartment in Pallikaranai with two bathrooms, lift access and community parking.", "broker_id": "BRK002", "image_folder": "images/demo/PROP004"},
        {"property_id": "PROP005", "property_type": "Villa", "bhk": 4, "price_lakhs": 210.0, "area_sqft": 2450, "location": "Madipakkam", "street": "Demo Garden Extension", "city": "Chennai", "pincode": "600091", "bathrooms": 4, "property_age": 3, "builder": "Demo Courtyard Developers", "description": "Synthetic demo 4 BHK villa in Madipakkam with four bathrooms, private garden space and a quiet residential setting.", "broker_id": "BRK002", "image_folder": "images/demo/PROP005"},
        {"property_id": "PROP006", "property_type": "Apartment", "bhk": 3, "price_lakhs": 185.0, "area_sqft": 1650, "location": "Anna Nagar", "street": "Demo Tower Road", "city": "Chennai", "pincode": "600040", "bathrooms": 3, "property_age": 6, "builder": "Demo Metro Homes", "description": "Synthetic demo 3 BHK apartment in Anna Nagar with three bathrooms, spacious living room and reserved parking.", "broker_id": "BRK003", "image_folder": "images/demo/PROP006"},
        {"property_id": "PROP007", "property_type": "Independent House", "bhk": 1, "price_lakhs": 52.0, "area_sqft": 720, "location": "Tambaram", "street": "Demo Station Link Road", "city": "Chennai", "pincode": "600045", "bathrooms": 1, "property_age": 10, "builder": "Demo Tambaram Homes", "description": "Synthetic demo 1 BHK independent house in Tambaram for a compact budget, with one bathroom and an accessible frontage.", "broker_id": "BRK004", "image_folder": "images/demo/PROP007"},
        {"property_id": "PROP008", "property_type": "Apartment", "bhk": 2, "price_lakhs": 98.0, "area_sqft": 1120, "location": "Perungudi", "street": "Demo Tech Corridor", "city": "Chennai", "pincode": "600096", "bathrooms": 2, "property_age": 1, "builder": "Demo Axis Living", "description": "Synthetic demo 2 BHK apartment in Perungudi near the IT corridor, with two bathrooms and new construction.", "broker_id": "BRK003", "image_folder": "images/demo/PROP008"},
        {"property_id": "PROP009", "property_type": "Villa", "bhk": 4, "price_lakhs": 240.0, "area_sqft": 2600, "location": "Sholinganallur", "street": "Demo OMR Garden Road", "city": "Chennai", "pincode": "600119", "bathrooms": 4, "property_age": 2, "builder": "Demo OMR Residences", "description": "Synthetic demo 4 BHK villa in Sholinganallur with four bathrooms, terrace space and access toward the OMR employment corridor.", "broker_id": "BRK003", "image_folder": "images/demo/PROP009"},
        {"property_id": "PROP010", "property_type": "Apartment", "bhk": 3, "price_lakhs": 110.0, "area_sqft": 1380, "location": "Medavakkam", "street": "Demo Greenway Avenue", "city": "Chennai", "pincode": "600100", "bathrooms": 3, "property_age": 3, "builder": "Demo Greenway Builders", "description": "Synthetic demo 3 BHK apartment in Medavakkam with three bathrooms, family-oriented amenities and two-wheeler parking.", "broker_id": "BRK002", "image_folder": "images/demo/PROP010"},
    ]

    land = [
        {"land_id": "LAND001", "land_type": "Residential Plot", "area_sqft": 1200, "price_lakhs": 72.0, "price_per_sqft": 6000, "location": "Velachery", "street": "Demo Lake View Road", "city": "Chennai", "pincode": "600042", "facing": "East", "broker_id": "BRK001", "description": "Synthetic demo east-facing residential plot in Velachery suitable for a compact independent home."},
        {"land_id": "LAND002", "land_type": "Residential Plot", "area_sqft": 1800, "price_lakhs": 108.0, "price_per_sqft": 6000, "location": "Adambakkam", "street": "Demo Lake Junction", "city": "Chennai", "pincode": "600088", "facing": "North", "broker_id": "BRK001", "description": "Synthetic demo north-facing residential plot in Adambakkam with dimensions suited to a family house."},
        {"land_id": "LAND003", "land_type": "Residential Plot", "area_sqft": 1500, "price_lakhs": 75.0, "price_per_sqft": 5000, "location": "Pallikaranai", "street": "Demo Marsh View Street", "city": "Chennai", "pincode": "600100", "facing": "West", "broker_id": "BRK002", "description": "Synthetic demo west-facing residential plot in Pallikaranai for a low-rise home project."},
        {"land_id": "LAND004", "land_type": "Villa Plot", "area_sqft": 2400, "price_lakhs": 132.0, "price_per_sqft": 5500, "location": "Madipakkam", "street": "Demo Garden Extension", "city": "Chennai", "pincode": "600091", "facing": "East", "broker_id": "BRK002", "description": "Synthetic demo east-facing villa plot in Madipakkam with space for a larger home and garden."},
        {"land_id": "LAND005", "land_type": "Residential Plot", "area_sqft": 1000, "price_lakhs": 68.0, "price_per_sqft": 6800, "location": "Anna Nagar", "street": "Demo Tower Road", "city": "Chennai", "pincode": "600040", "facing": "South", "broker_id": "BRK003", "description": "Synthetic demo south-facing residential plot in Anna Nagar intended for a compact urban home."},
        {"land_id": "LAND006", "land_type": "Residential Plot", "area_sqft": 2000, "price_lakhs": 90.0, "price_per_sqft": 4500, "location": "Tambaram", "street": "Demo Station Link Road", "city": "Chennai", "pincode": "600045", "facing": "North-East", "broker_id": "BRK004", "description": "Synthetic demo north-east-facing residential plot in Tambaram with room for a duplex plan."},
        {"land_id": "LAND007", "land_type": "Mixed-use Plot", "area_sqft": 2200, "price_lakhs": 176.0, "price_per_sqft": 8000, "location": "Perungudi", "street": "Demo Tech Corridor", "city": "Chennai", "pincode": "600096", "facing": "East", "broker_id": "BRK003", "description": "Synthetic demo east-facing mixed-use plot in Perungudi near the business corridor; verify permissions before purchase."},
        {"land_id": "LAND008", "land_type": "Residential Plot", "area_sqft": 1600, "price_lakhs": 88.0, "price_per_sqft": 5500, "location": "Medavakkam", "street": "Demo Greenway Avenue", "city": "Chennai", "pincode": "600100", "facing": "West", "broker_id": "BRK002", "description": "Synthetic demo west-facing residential plot in Medavakkam for a family home with parking."},
    ]

    def amenity_rows(kind, name, description):
        return [
            {f"{kind}_id": f"{kind[:3].upper()}{index:03d}", f"{kind}_name": f"Demo {name} {area}", "area": area, "street": f"Demo {area} Main Road", "address": f"Synthetic demo {name.lower()} address, {area}", "city": "Chennai", "pincode": location["pincode"], f"{kind}_type": f"Demo {name} Type", "description": f"Synthetic demo record: {description} serving {area}. Verify current availability and services independently."}
            for index, (area, location) in enumerate(zip([row["area"] for row in locations], locations), 1)
        ]

    schools = amenity_rows("school", "School", "local school reference")
    colleges = amenity_rows("college", "College", "local college reference")
    hospitals = amenity_rows("hospital", "Hospital", "local hospital reference")
    supermarkets = [
        {"supermarket_id": f"SUP{index:03d}", "supermarket_name": f"Demo Market {location['area']}", "area": location["area"], "street": f"Demo {location['area']} Main Road", "address": f"Synthetic demo supermarket address, {location['area']}", "city": "Chennai", "pincode": location["pincode"], "description": f"Synthetic demo supermarket reference serving {location['area']}. Verify current hours and stock independently."}
        for index, location in enumerate(locations, 1)
    ]
    transport_types = ["Bus Stop", "Railway Station", "Metro Station", "Auto Stand"]
    transport = [
        {"transport_id": f"TRN{index:03d}", "transport_name": f"Demo Transit Point {location['area']}", "transport_type": transport_types[(index - 1) % len(transport_types)], "area": location["area"], "street": f"Demo {location['area']} Main Road", "address": f"Synthetic demo transport address, {location['area']}", "city": "Chennai", "pincode": location["pincode"], "description": f"Synthetic demo {transport_types[(index - 1) % len(transport_types)].lower()} reference for {location['area']}. Verify routes and operating status independently."}
        for index, location in enumerate(locations, 1)
    ]

    location_coordinates = {row["area"].lower(): (row["latitude"], row["longitude"]) for row in locations}

    def add_demo_coordinates(rows, area_column, latitude_step, longitude_step):
        for index, row in enumerate(rows, 1):
            latitude, longitude = location_coordinates[row[area_column].lower()]
            row["latitude"] = round(latitude + (index % 4) * latitude_step, 6)
            row["longitude"] = round(longitude + (index % 5) * longitude_step, 6)
        return rows

    add_demo_coordinates(properties, "location", 0.0002, 0.0002)
    add_demo_coordinates(land, "location", 0.0003, 0.0003)
    add_demo_coordinates(schools, "area", 0.0004, 0.0003)
    add_demo_coordinates(colleges, "area", 0.0005, 0.0004)
    add_demo_coordinates(hospitals, "area", 0.0006, 0.0005)
    add_demo_coordinates(supermarkets, "area", 0.0007, 0.0006)
    add_demo_coordinates(transport, "area", 0.0008, 0.0007)

    return {
        "properties.csv": properties,
        "land.csv": land,
        "schools.csv": schools,
        "colleges.csv": colleges,
        "hospitals.csv": hospitals,
        "supermarkets.csv": supermarkets,
        "transport.csv": transport,
        "locations.csv": locations,
        "brokers.csv": brokers,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = build_datasets()
    for filename, rows in datasets.items():
        pd.DataFrame(rows).to_csv(DATA_DIR / filename, index=False)
        print(f"{filename}: {len(rows)} records")
    print(f"Generated {len(datasets)} datasets in {DATA_DIR}")


if __name__ == "__main__":
    main()