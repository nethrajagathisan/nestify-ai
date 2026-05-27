"""
Load and normalize Kaggle Indian real estate data into Nestify-AI's listing schema.

Supported dataset:
  "Housing Prices in Metropolitan Areas of India" by Ruchi Bhatia
  https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india

  Columns in this dataset:
    Price, Area, Location, No. of Bedrooms, Resale,
    + 30 binary amenity columns (Gymnasium, SwimmingPool, CarParking, LiftAvailable, ...)

Usage:
    python scripts/load_kaggle_data.py --data-dir ./kaggle_data
    python scripts/load_kaggle_data.py --data-dir ./kaggle_data --per-city 50
    python scripts/load_kaggle_data.py --data-dir ./kaggle_data --limit 20   # quick test
"""

import argparse
import json
import os
import random
import re
import sys
import uuid
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Run: pip install pandas")
    sys.exit(1)


random.seed(99)

# ── City mapping: CSV filename stem → canonical city name ────────────────────
CITY_FILE_MAP = {
    "bangalore": "Bangalore",
    "bengaluru": "Bangalore",
    "chennai":   "Chennai",
    "delhi":     "Delhi",
    "hyderabad": "Hyderabad",
    "mumbai":    "Mumbai",
    "pune":      "Pune",
    "coimbatore":"Coimbatore",
}

# ── Amenity binary columns → readable names ──────────────────────────────────
AMENITY_COL_MAP = {
    "gymnasium":          "gym",
    "swimmingpool":       "swimming pool",
    "landscapedgardens":  "landscaped gardens",
    "joggingtrack":       "jogging track",
    "rainwaterharvesting":"rainwater harvesting",
    "indoorgames":        "indoor games",
    "clubhouse":          "club house",
    "24x7security":       "security",
    "powerbackup":        "power backup",
    "carparking":         "parking",
    "liftavailable":      "lift",
    "sportsfacility":     "sports facility",
    "intercom":           "intercom",
    "ac":                 "air conditioner",
    "wifi":               "wifi",
    "washingmachine":     "washing machine",
    "wardrobe":           "wardrobe",
    "gasconnection":      "gas connection",
    "cafeteria":          "cafeteria",
    "multipurposeroom":   "multipurpose room",
    "shoppingmall":       "shopping mall nearby",
    "school":             "school nearby",
    "hospital":           "hospital nearby",
    "atm":                "ATM nearby",
}

FACING_OPTIONS = ["East", "West", "North", "South", "North-East", "North-West", "South-East", "South-West"]

CITY_BUILDERS = {
    "Bangalore":  ["Prestige", "Brigade", "Sobha", "Puravankara", "Godrej"],
    "Chennai":    ["Prestige", "Appaswamy", "Casagrand", "Akshaya", "L&T"],
    "Delhi":      ["DLF", "Unitech", "Ansal", "Sobha", "Tata Housing"],
    "Hyderabad":  ["My Home", "Aparna", "Ramky", "Lodha", "Sumadhura"],
    "Mumbai":     ["Lodha", "Oberoi", "Hiranandani", "Godrej", "Rustomjee"],
    "Pune":       ["Kolte Patil", "Vilas Javdekar", "Godrej", "Mahindra", "Nyati"],
    "Coimbatore": ["Provident", "Salarpuria", "Assetz", "Casagrand", "Sobha"],
}

CITY_LOCALITY_FALLBACKS = {
    "Bangalore":  ["Koramangala", "Whitefield", "Indiranagar", "HSR Layout", "Marathahalli"],
    "Chennai":    ["OMR", "Anna Nagar", "Adyar", "T Nagar", "Velachery"],
    "Delhi":      ["Vasant Kunj", "Dwarka", "Rohini", "Saket", "Karol Bagh"],
    "Hyderabad":  ["Gachibowli", "Hitech City", "Banjara Hills", "Kondapur", "Madhapur"],
    "Mumbai":     ["Bandra", "Andheri", "Powai", "Thane", "Malad"],
    "Pune":       ["Koregaon Park", "Hinjewadi", "Baner", "Kharadi", "Hadapsar"],
    "Coimbatore": ["RS Puram", "Peelamedu", "Race Course", "Saibaba Colony", "Singanallur"],
}

CITY_TAGS = {
    "Bangalore":  ["IT hub", "startup culture", "pleasant climate", "metro connectivity"],
    "Chennai":    ["coastal city", "cultural heritage", "automotive hub", "traditional markets"],
    "Delhi":      ["capital city", "monuments nearby", "metro connectivity", "wide roads"],
    "Hyderabad":  ["pharma hub", "historic charm", "lake views", "metro expansion"],
    "Mumbai":     ["financial capital", "coastal living", "local train network", "sea views"],
    "Pune":       ["education hub", "pleasant weather", "IT parks", "hill views"],
    "Coimbatore": ["textile city", "Western Ghats views", "peaceful living", "emerging IT"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _price_to_lakhs(val) -> float | None:
    if pd.isna(val):
        return None
    try:
        n = float(val)
    except (ValueError, TypeError):
        return None
    # Values > 1,00,000 are in raw INR
    if n > 100_000:
        return round(n / 100_000, 2)
    return round(n, 2)


def _infer_bhk(val) -> int:
    if pd.isna(val):
        return 2
    try:
        n = int(float(str(val)))
        return n if 1 <= n <= 4 else 2
    except (ValueError, TypeError):
        return 2


def _clean_locality(val, city: str) -> str:
    if pd.isna(val) or not str(val).strip():
        return random.choice(CITY_LOCALITY_FALLBACKS.get(city, [city]))
    # Strip extra route descriptors like "on Tumkur Road" that make it verbose
    loc = str(val).strip()
    loc = re.sub(r"\s+on\s+.+", "", loc, flags=re.IGNORECASE).strip()
    return loc.title() if loc else random.choice(CITY_LOCALITY_FALLBACKS.get(city, [city]))


def _extract_amenities(row: pd.Series) -> list[str]:
    """Read binary amenity columns and return a list of present amenities."""
    amenities = []
    for col_raw, name in AMENITY_COL_MAP.items():
        # Match column name case-insensitively, ignoring spaces/punctuation
        for col in row.index:
            col_norm = re.sub(r"[^a-z0-9]", "", col.lower())
            if col_norm == col_raw:
                try:
                    if int(float(row[col])) == 1:
                        amenities.append(name)
                except (ValueError, TypeError):
                    pass
                break
    return amenities


def _build_description(bhk: int, locality: str, city: str, area: int,
                        year: int, facing: str, amenities: list[str]) -> str:
    tags = CITY_TAGS.get(city, ["great connectivity", "modern amenities"])
    tag1, tag2 = random.sample(tags, 2)
    top_amenities = [a for a in amenities if a in ("gym", "swimming pool", "club house", "landscaped gardens")]
    amenity_str = f" Features include {', '.join(top_amenities[:2])}." if top_amenities else ""
    return (
        f"{bhk}BHK apartment in {locality}, {city}, built in {year}. "
        f"Spans {area} sq.ft with {facing}-facing orientation and modern fittings."
        f"{amenity_str} "
        f"Located in a {tag1} neighbourhood with {tag2}."
    )


def _build_title(bhk: int, locality: str, city: str, is_resale: bool) -> str:
    prefix = "Resale" if is_resale else "Premium"
    return f"{prefix} {bhk}BHK Apartment in {locality}, {city}"


def _floor_info() -> tuple[str, int]:
    total = random.choice([5, 8, 10, 12, 14, 16, 20, 24])
    floor_num = random.randint(0, total)
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    if floor_num == 0:
        floor_str = "Ground"
    else:
        suffix = suffixes.get(floor_num if floor_num <= 3 else 0, "th")
        floor_str = f"{floor_num}{suffix}"
    return floor_str, total


# ── Per-file processor ────────────────────────────────────────────────────────

def process_csv(csv_path: Path, city: str, limit: int | None = None) -> list[dict]:
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")

    # Normalise column names for lookup (keep originals for amenity extraction)
    col_lower = {c.lower().replace(" ", "").replace(".", ""): c for c in df.columns}

    price_col    = col_lower.get("price")
    area_col     = col_lower.get("area")
    location_col = col_lower.get("location") or col_lower.get("locality") or col_lower.get("address")
    bhk_col      = col_lower.get("noofbedrooms") or col_lower.get("bhk") or col_lower.get("bedroom")
    resale_col   = col_lower.get("resale")

    if not price_col or not area_col:
        print(f"  [SKIP] {csv_path.name} — could not find Price/Area columns. Found: {list(df.columns)}")
        return []

    if limit:
        df = df.head(limit)

    records, skipped = [], 0

    for _, row in df.iterrows():
        price = _price_to_lakhs(row.get(price_col))
        area_raw = row.get(area_col)
        try:
            area = int(float(area_raw)) if not pd.isna(area_raw) else None
        except (ValueError, TypeError):
            area = None

        if not price or not area or price < 5 or area < 100:
            skipped += 1
            continue

        bhk = _infer_bhk(row.get(bhk_col)) if bhk_col else 2
        locality = _clean_locality(row.get(location_col) if location_col else None, city)

        is_resale = False
        if resale_col and not pd.isna(row.get(resale_col)):
            try:
                is_resale = int(float(row[resale_col])) == 1
            except (ValueError, TypeError):
                pass

        amenities = _extract_amenities(row)
        has_parking = "parking" in amenities
        facing = random.choice(FACING_OPTIONS)
        year_built = random.randint(2010, 2024)
        builder = random.choice(CITY_BUILDERS.get(city, ["Premium Builders"]))
        floor, total_floors = _floor_info()

        records.append({
            "id": f"prop_{str(uuid.uuid4())[:8]}",
            "title": _build_title(bhk, locality, city, is_resale),
            "city": city,
            "locality": locality,
            "bhk": bhk,
            "price_lakhs": price,
            "area_sqft": area,
            "property_type": "apartment",
            "amenities": amenities,
            "description": _build_description(bhk, locality, city, area, year_built, facing, amenities),
            "year_built": year_built,
            "parking": has_parking,
            "builder": builder,
            "facing": facing,
            "floor": floor,
            "total_floors": total_floors,
        })

    print(f"  {csv_path.name}: {len(records)} loaded, {skipped} skipped")
    return records


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load Kaggle Indian real estate data into Nestify-AI format")
    parser.add_argument("--data-dir", required=True, help="Folder containing Kaggle CSV files")
    parser.add_argument("--output", default=os.path.join("data", "properties", "listings.json"))
    parser.add_argument("--limit", type=int, default=None, help="Max rows to read per CSV (for testing)")
    parser.add_argument("--per-city", type=int, default=40, help="Max listings to keep per city (default 40)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: --data-dir '{data_dir}' does not exist.")
        sys.exit(1)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in '{data_dir}'")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} CSV file(s)\n")
    all_listings: list[dict] = []

    for csv_path in csv_files:
        city = CITY_FILE_MAP.get(csv_path.stem.lower())
        if not city:
            print(f"  [SKIP] {csv_path.name} — not a supported city")
            continue
        print(f"Processing {csv_path.name} -> {city}")
        records = process_csv(csv_path, city, limit=args.limit)
        random.shuffle(records)
        all_listings.extend(records[:args.per_city])

    if not all_listings:
        print("\nERROR: No listings produced.")
        sys.exit(1)

    random.shuffle(all_listings)
    for i, listing in enumerate(all_listings, start=1):
        listing["id"] = f"prop_{i:03d}"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_listings, f, indent=2, ensure_ascii=False)

    city_counts: dict[str, int] = {}
    for l in all_listings:
        city_counts[l["city"]] = city_counts.get(l["city"], 0) + 1

    print(f"\nSaved {len(all_listings)} listings -> {output_path}")
    print("\nBreakdown by city:")
    for city, count in sorted(city_counts.items()):
        print(f"  {city:15s}  {count} listings")


if __name__ == "__main__":
    main()
