import os
import pandas as pd

def test_dataset():
    """
    Validates and displays summary statistics of the cleaned Chennai real estate dataset.
    Used to verify data integrity before document creation and embedding.
    """
    dataset_path = os.path.join("data", "cleaned_dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join("data", "chennai_house_price_cleaned.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {dataset_path}! Please run clean_dataset.py first.")

    df = pd.read_csv(dataset_path)
    print("=" * 50)
    print("REAL ESTATE DATASET VALIDATION REPORT")
    print("=" * 50)
    print(f"File: {dataset_path}")
    print(f"Total Properties: {len(df)}")
    print(f"Total Features: {len(df.columns)}")
    print(f"Columns: {df.columns.tolist()}")

    print("\n--- Missing Value Check ---")
    missing = df.isnull().sum()
    print(missing)
    assert missing.sum() == 0, "Dataset contains unexpected missing values!"
    print(">> Validation: 0 missing values across all columns. PASS.")

    print("\n--- Numeric Ranges ---")
    print(f"BHK types: {sorted(df['bhk'].unique().tolist())}")
    print(f"Price range: Rs. {df['price'].min():.2f} Lakhs to Rs. {df['price'].max():.2f} Lakhs")
    print(f"Area range: {df['area'].min():.0f} sq.ft to {df['area'].max():.0f} sq.ft")
    print(f"Bathroom values: {sorted(df['bathroom'].unique().tolist())}")

    print("\n--- Construction Status Distribution ---")
    print(df["status"].value_counts().to_string())

    print("\n--- Top 10 Locations in Chennai ---")
    print(df["location"].value_counts().head(10).to_string())

    print("\n--- Top 5 Builders ---")
    print(df["builder"].value_counts().head(5).to_string())

    print("\n" + "=" * 50)
    print("ALL DATA INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    test_dataset()