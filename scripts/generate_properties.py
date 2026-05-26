"""Generate 150 realistic Indian real estate property listings."""

import json
import random
import os

random.seed(42)

CITY_CONFIG = {
    "Bangalore": {
        "count": 30,
        "min_price": 50,
        "max_price": 300,
        "localities": ["Koramangala", "Whitefield", "Indiranagar", "HSR Layout", "Electronic City",
                       "JP Nagar", "Marathahalli", "Bellandur", "Sarjapur Road", "Hebbal",
                       "BTM Layout", "Rajajinagar", "Malleshwaram", "Basavanagudi", "Yelahanka"],
        "tags": ["IT hub", "startup culture", "pleasant climate", "green spaces", "metro connectivity"],
        "builders": ["Prestige", "Brigade", "Sobha", "Puravankara", "Godrej", "DLF"],
    },
    "Chennai": {
        "count": 25,
        "min_price": 40,
        "max_price": 250,
        "localities": ["OMR", "Anna Nagar", "Adyar", "T Nagar", "Velachery",
                       "Kodambakkam", "Porur", "Madipakkam", "Perungudi", "Pallikaranai",
                       "Sholinganallur", "Guindy", "Mylapore", "Nungambakkam", "Besant Nagar"],
        "tags": ["coastal city", "cultural heritage", "automotive hub", "BEACH access", "traditional markets"],
        "builders": ["Prestige", "Brigade", "Appaswamy", "Casagrand", "Akshaya", "L&T"],
    },
    "Hyderabad": {
        "count": 25,
        "min_price": 35,
        "max_price": 200,
        "localities": ["Gachibowli", "Hitech City", "Banjara Hills", "Jubilee Hills", "Kondapur",
                       "Madhapur", "Secunderabad", "Kukatpally", "Manikonda", "Nallagandla",
                       "Kompally", "Miyapur", "Tellapur", "Kokapet", "Shamshabad"],
        "tags": ["pharma hub", "historic charm", "biryani capital", "lake views", "metro expansion"],
        "builders": ["My Home", "Aparna", "Ramky", "Lodha", "Sumadhura", "Prajay"],
    },
    "Mumbai": {
        "count": 25,
        "min_price": 80,
        "max_price": 500,
        "localities": ["Bandra", "Andheri", "Powai", "Worli", "Thane",
                       "Malad", "Goregaon", "Vikhroli", "Chembur", "Kandivali",
                       "Borivali", "Dadar", "Lower Parel", "Bhandup", "Airoli"],
        "tags": ["financial capital", "coastal living", "Bollywood glamour", "local train network", "sea views"],
        "builders": ["Lodha", "Oberoi", "Hiranandani", "Godrej", "Rustomjee", "Runwal"],
    },
    "Pune": {
        "count": 20,
        "min_price": 35,
        "max_price": 180,
        "localities": ["Koregaon Park", "Kalyani Nagar", "Viman Nagar", "Hinjewadi", "Wakad",
                       "Baner", "Aundh", "Magarpatta", "Kharadi", "Hadapsar",
                       "Pimple Saudagar", "Balewadi", "Kondhwa", "Sus", "Wagholi"],
        "tags": ["education hub", "pleasant weather", "IT parks", "hill views", "youthful vibe"],
        "builders": ["Kolte Patil", "Vilas Javdekar", "Godrej", "Lodha", "Mahindra", "Nyati"],
    },
    "Delhi": {
        "count": 15,
        "min_price": 60,
        "max_price": 350,
        "localities": ["Vasant Kunj", "Greater Kailash", "Dwarka", "Rohini", "Saket",
                       "Janakpuri", "Lajpat Nagar", "Pitampura", "Shahdara", "Karol Bagh",
                       "Rajouri Garden", "Mayur Vihar", "Malviya Nagar", "South Extension", "Ashok Vihar"],
        "tags": ["capital city", "monuments nearby", "metro dense", "political hub", "wide roads"],
        "builders": ["DLF", "Unitech", "Ansal", "Sobha", "Tata Housing", "IREO"],
    },
    "Coimbatore": {
        "count": 10,
        "min_price": 20,
        "max_price": 120,
        "localities": ["RS Puram", "Peelamedu", "Race Course", "Saibaba Colony", "Singanallur",
                       "Ganapathy", "Saravanampatti", "Kuniyamuthur", "Ukkadam", "Ramanathapuram"],
        "tags": ["textile city", "Western Ghats views", "peaceful living", "emerging IT", "temple town"],
        "builders": ["Provident", "Salarpuria", "Assetz", "Casagrand", "Sobha", "Puravankara"],
    },
}

ADJECTIVES = ["Spacious", "Modern", "Luxurious", "Premium", "Elegant", "Contemporary",
              "Well-Ventilated", "Bright", "Stylish", "Designer", "Fully-Furnished",
              "Semi-Furnished", "Newly-Built", "Ready-to-Move", "Gated", "Green",
              "Smart", "Eco-Friendly", "High-Rise", "Boutique"]

AMENITIES_POOL = ["gym", "swimming pool", "parking", "security", "lift", "club house", "power backup", "rainwater harvesting"]

PROPERTY_TYPES = ["apartment", "apartment", "apartment", "apartment", "apartment", "villa", "villa", "plot", "commercial"]

BHK_WEIGHTS = [1, 2, 2, 3, 3, 3, 4]

FACING = ["East", "West", "North", "South", "North-East", "North-West", "South-East", "South-West"]

FLOOR_TEMPLATE = ["Ground", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", "15th", "18th", "20th", "25th"]


def area_for_bhk_city(bhk, city):
    base = {
        1: (450, 650),
        2: (750, 1100),
        3: (1200, 1800),
        4: (1800, 3000),
    }
    low, high = base[bhk]
    if city == "Mumbai":
        low, high = int(low * 0.85), int(high * 0.9)
    elif city == "Coimbatore":
        low, high = int(low * 1.1), int(high * 1.15)
    return random.randint(low, high)


def generate_listing(idx, city):
    cfg = CITY_CONFIG[city]
    bhk = random.choice(BHK_WEIGHTS)
    area = area_for_bhk_city(bhk, city)
    locality = random.choice(cfg["localities"])
    builder = random.choice(cfg["builders"])
    adj = random.choice(ADJECTIVES)
    ptype = random.choice(PROPERTY_TYPES)

    if ptype == "plot":
        title = f"{adj} Residential Plot in {locality}, {city}"
        bhk = 0
        area = random.randint(1200, 4000)
    elif ptype == "commercial":
        title = f"{adj} Commercial Space in {locality}, {city}"
        bhk = 0
        area = random.randint(400, 2500)
    else:
        title = f"{adj} {bhk}BHK {ptype.title()} in {locality}, {city}"

    # Price logic
    if ptype == "plot":
        price = round(random.uniform(cfg["min_price"] * 0.6, cfg["max_price"] * 0.5), 2)
    elif ptype == "commercial":
        price = round(random.uniform(cfg["min_price"] * 0.8, cfg["max_price"] * 0.7), 2)
    else:
        price = round(random.uniform(cfg["min_price"], cfg["max_price"]), 2)

    price = max(10.0, round(price, 2))

    # Amenities
    num_amenities = random.randint(3, 6)
    amenities = random.sample(AMENITIES_POOL, num_amenities)

    # Description
    tag1, tag2 = random.sample(cfg["tags"], 2)
    facing = random.choice(FACING)
    year = random.randint(2010, 2024)

    if ptype == "plot":
        desc = (f"Prime {area} sq.ft residential plot in {locality}, {city}. "
                f"Perfect for building your dream home with {facing}-facing orientation. "
                f"Located in a {tag1} area with {tag2} advantages.")
    elif ptype == "commercial":
        desc = (f"{adj} commercial property in {locality}, {city}. "
                f"Ideal for offices or retail with {facing} facing and excellent connectivity. "
                f"Situated in a {tag1} zone known for {tag2}.")
    else:
        desc = (f"{adj} {bhk}BHK {ptype} in {locality}, {city}, built in {year}. "
                f"Offers {area} sq.ft of {facing}-facing living space with modern fittings. "
                f"Located in a {tag1} neighborhood with {tag2}.")

    parking = random.choice([True, True, True, False]) if ptype != "plot" else False

    if ptype in ["apartment", "villa"]:
        total_floors = random.choice([4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30])
        if ptype == "villa":
            floor = "Ground + 1"
        else:
            floor = random.choice(FLOOR_TEMPLATE[:min(total_floors, len(FLOOR_TEMPLATE))])
    else:
        total_floors = 0
        floor = ""

    return {
        "id": f"prop_{idx:03d}",
        "title": title,
        "city": city,
        "locality": locality,
        "bhk": bhk,
        "price_lakhs": price,
        "area_sqft": area,
        "property_type": ptype,
        "amenities": amenities,
        "description": desc,
        "year_built": year if ptype in ["apartment", "villa"] else 0,
        "parking": parking,
        "builder": builder,
        "facing": facing,
        "floor": floor,
        "total_floors": total_floors,
    }


def main():
    listings = []
    idx = 1
    for city, cfg in CITY_CONFIG.items():
        for _ in range(cfg["count"]):
            listings.append(generate_listing(idx, city))
            idx += 1

    random.shuffle(listings)
    # Re-assign sequential IDs after shuffle
    for i, lst in enumerate(listings, start=1):
        lst["id"] = f"prop_{i:03d}"

    output_dir = os.path.join("data", "properties")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "listings.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(listings, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(listings)} listings -> {output_path}")


if __name__ == "__main__":
    main()
