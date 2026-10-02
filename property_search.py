import os
import pandas as pd

def filter_properties(location=None, bhk=None, max_price=None, min_price=None, status=None, dataset_path=None):
    """
    Filters the cleaned Chennai property dataset using structured criteria.
    Useful for exact/tabular property lookups and verification.
    """
    if dataset_path is None:
        dataset_path = os.path.join("data", "cleaned_dataset.csv")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join("data", "chennai_house_price_cleaned.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {dataset_path}! Run clean_dataset.py first.")

    df = pd.read_csv(dataset_path)
    results = df.copy()

    if location:
        results = results[results["location"].astype(str).str.lower().str.contains(location.strip().lower(), na=False)]

    if bhk is not None:
        results = results[results["bhk"] == int(bhk)]

    if max_price is not None:
        results = results[results["price"] <= float(max_price)]

    if min_price is not None:
        results = results[results["price"] >= float(min_price)]

    if status:
        results = results[results["status"].astype(str).str.lower().str.contains(status.strip().lower(), na=False)]

    return results

def main():
    print("=" * 60)
    print("Structured Chennai Property Search")
    print("=" * 60)

    try:
        loc_input = input("Enter location (e.g., Anna Nagar, Velachery, or leave blank for all): ").strip()
        location = loc_input if loc_input else None

        bhk_input = input("Enter number of BHK (e.g., 2, 3, or leave blank): ").strip()
        bhk = int(bhk_input) if bhk_input.isdigit() else None

        price_input = input("Enter maximum price in lakhs (e.g., 60, or leave blank): ").strip()
        max_price = float(price_input) if price_input else None

        results = filter_properties(location=location, bhk=bhk, max_price=max_price)

        print(f"\nProperties found: {len(results)}")
        if len(results) > 0:
            print("\nMatching properties (Top 10):")
            cols = ["price", "area", "status", "bhk", "bathroom", "age", "location", "builder"]
            print(results[cols].head(10).to_string(index=False))
        else:
            print("Sorry, no matching properties found for the specified criteria.")

    except Exception as e:
        print(f"Error during property search: {e}")

if __name__ == "__main__":
    main()