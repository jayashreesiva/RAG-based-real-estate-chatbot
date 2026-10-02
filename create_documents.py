import os
import pandas as pd

def create_property_documents():
    """
    Transforms structured tabular records from cleaned_dataset.csv into
    rich, natural-language document chunks optimized for semantic embedding and retrieval.
    Saves generated documents into data/property_documents.txt.
    """
    input_path = os.path.join("data", "cleaned_dataset.csv")
    output_path = os.path.join("data", "property_documents.txt")

    if not os.path.exists(input_path):
        input_path = os.path.join("data", "chennai_house_price_cleaned.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {input_path}! Run clean_dataset.py first.")

    print(f"Reading cleaned dataset from: {input_path}")
    df = pd.read_csv(input_path)

    documents = []

    for index, row in df.iterrows():
        prop_id = index + 1
        price = float(row["price"])
        area = int(row["area"])
        status = str(row["status"]).strip()
        bhk = int(row["bhk"])
        bathroom = float(row["bathroom"])
        age = float(row["age"])
        location = str(row["location"]).strip()
        builder = str(row["builder"]).strip()

        bathroom_text = f"{int(bathroom)} Bathrooms" if bathroom > 0 else "Not specified"
        age_text = f"{int(age)} years" if age > 0 else "New / Under Construction"

        doc = f"""Property ID: {prop_id}
Location: {location}, Chennai
BHK: {bhk} BHK
Price: {price:.2f} lakhs
Area: {area} sq.ft
Status: {status}
Bathroom: {bathroom_text}
Age: {age_text}
Builder: {builder}
Summary: {bhk} BHK residential property located in {location}, Chennai with an area of {area} sq.ft, priced at {price:.2f} lakhs. Status is {status}. Property age is {age_text} with {bathroom_text}, built by {builder}."""

        documents.append(doc.strip())

    # Save documents separated by standard delimiter
    with open(output_path, "w", encoding="utf-8") as file:
        for doc in documents:
            file.write(doc)
            file.write("\n\n-----------------------------\n\n")

    # Clean up redundant root copy if present
    root_doc_path = "property_documents.txt"
    if os.path.exists(root_doc_path):
        try:
            os.remove(root_doc_path)
            print(f"Removed redundant root copy: {root_doc_path}")
        except Exception as e:
            print(f"Note: Could not remove root {root_doc_path}: {e}")

    print(f"\nProperty documents created successfully!")
    print(f"Total documents generated: {len(documents)}")
    print(f"Saved to: {output_path}")

    # Display sample document
    print("\n--- Sample Document (ID: 1) ---")
    print(documents[0])
    print("--------------------------------")

    return documents

if __name__ == "__main__":
    create_property_documents()