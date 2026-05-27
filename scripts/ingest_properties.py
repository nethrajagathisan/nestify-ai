import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.ingestion.property_loader import load_properties

if __name__ == "__main__":
    try:
        load_properties()
    except Exception as e:
        print(f"Error during property ingestion: {e}")
        sys.exit(1)
