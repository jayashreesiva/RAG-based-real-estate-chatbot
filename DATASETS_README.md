# Chennai Demo Dataset Phase

These datasets extend the existing Chennai real-estate project without changing its current RAG files. Every new record is synthetic demo data. Property and broker contact values are fictional, and location coordinates are synthetic demo coordinates; verify any real-world information independently before use.

## Files

- `properties.csv`: demo apartments, independent houses and villas. `price_lakhs` is in Indian lakhs; `image_folder` is a future relative image-folder placeholder; `latitude` and `longitude` are synthetic demo coordinates.
- `land.csv`: demo residential, villa and mixed-use plots. `price_per_sqft` is in rupees per square foot; coordinates are synthetic demo coordinates.
- `schools.csv`, `colleges.csv`, `hospitals.csv`: synthetic nearby-amenity references with type, description and synthetic demo coordinate fields.
- `supermarkets.csv`: synthetic supermarket references with synthetic demo coordinates.
- `transport.csv`: synthetic bus stops, railway stations, metro stations and auto stands with synthetic demo coordinates.
- `locations.csv`: area lookup records, synthetic latitude/longitude values and logically connected nearby areas.
- `brokers.csv`: fictional demo broker contacts. Emails use `example.com` and phone numbers use the reserved-looking `+91-90000-000xx` pattern.

## Connections

`properties.csv` and `land.csv` connect to `brokers.csv` through `broker_id`, and both connect to `locations.csv` through `location`/`area`. The school, college, hospital, supermarket and transport datasets connect to `locations.csv` through `area`. All records use Chennai and the same controlled area vocabulary.

## Generate and validate

From the project root, run:

```powershell
python generate_datasets.py
python validate_datasets.py
```

The generator uses deterministic in-code records, requires no internet connection, uses pandas, preserves existing files, and prints a record count for each generated CSV. The validator checks file readability, schemas, IDs, non-empty fields, numeric values, area consistency and broker references. A successful validation prints `DATASET VALIDATION PASSED`.

The existing files in `data/` are legacy project artifacts and are intentionally preserved. These scripts write only the nine dataset filenames listed above.