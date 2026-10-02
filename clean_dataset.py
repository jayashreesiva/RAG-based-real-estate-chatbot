import os
import shutil
import pandas as pd

def clean_real_estate_dataset():
    """
    Cleans and standardizes the Chennai real estate dataset.
    Ensures data consistency, handles missing values, and prepares
    clean data for document chunking and vector embedding.
    """
    os.makedirs("data", exist_ok=True)

    original_path = os.path.join("data", "original_dataset.csv")
    legacy_path = os.path.join("data", "clean_data.csv")
    output_path = os.path.join("data", "cleaned_dataset.csv")
    legacy_cleaned_path = os.path.join("data", "chennai_house_price_cleaned.csv")

    # Ensure data/original_dataset.csv exists
    if not os.path.exists(original_path):
        if os.path.exists(legacy_path):
            shutil.copyfile(legacy_path, original_path)
            print(f"Copied {legacy_path} -> {original_path}")
        else:
            raise FileNotFoundError(f"Neither {original_path} nor {legacy_path} was found!")

    # Load dataset
    print(f"Loading raw dataset from: {original_path}")
    df = pd.read_csv(original_path)
    initial_shape = df.shape
    print(f"Initial dataset shape: {initial_shape[0]} rows, {initial_shape[1]} columns")

    # Normalize column names: strip whitespace and lowercase
    df.columns = df.columns.str.strip().str.lower()
    print("Columns identified:", list(df.columns))

    # Strip whitespace in string columns
    str_cols = ["status", "location", "builder"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Clean up unknown/nan strings
            df[col] = df[col].replace({"nan": "Unknown", "": "Unknown"})

    # Clean numeric fields
    numeric_cols = ["price", "area", "bhk", "bathroom", "age"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing values:
    # Bathroom: fill missing with 0 (or median where appropriate, 0 indicates unspecified/standard)
    df["bathroom"] = df["bathroom"].fillna(0).astype(float)

    # Age: fill missing with 0 (new / ready / under construction)
    df["age"] = df["age"].fillna(0).astype(float)

    # Price and Area: remove rows where price or area is missing or <= 0
    df = df.dropna(subset=["price", "area", "bhk"])
    df = df[(df["price"] > 0) & (df["area"] > 0) & (df["bhk"] > 0)]

    # Format status standard casing
    if "status" in df.columns:
        df["status"] = df["status"].apply(lambda s: s.title() if isinstance(s, str) else "Ready To Move")

    # Standardize location casing
    if "location" in df.columns:
        df["location"] = df["location"].apply(lambda l: l.title() if isinstance(l, str) else "Chennai")

    # Drop duplicate rows if any
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate rows...")
        df = df.drop_duplicates().reset_index(drop=True)

    final_shape = df.shape
    print(f"\nMissing values after cleaning:")
    print(df.isnull().sum())

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    # Also save to legacy path for backward compatibility
    df.to_csv(legacy_cleaned_path, index=False)

    print(f"\nCleaned dataset saved successfully to: {output_path}")
    print(f"Backward-compatible copy saved to: {legacy_cleaned_path}")
    print(f"Final clean dataset shape: {final_shape[0]} rows, {final_shape[1]} columns")
    print(f"Price range: Rs. {df['price'].min():.2f} Lakhs to Rs. {df['price'].max():.2f} Lakhs")
    print(f"Total unique locations in Chennai: {df['location'].nunique()}")

    return df

if __name__ == "__main__":
    clean_real_estate_dataset()