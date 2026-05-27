import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.ingestion.faq_loader import load_faq_docs


if __name__ == "__main__":
    try:
        load_faq_docs()
    except Exception as e:
        print(f"Error during FAQ ingestion: {e}")
        sys.exit(1)
