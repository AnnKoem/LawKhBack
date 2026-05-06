from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag"))

from law_assets import build_law_index, prepare_assets_from_source


if __name__ == "__main__":
    prepared = prepare_assets_from_source()
    index = build_law_index()
    print(prepared)
    print(f"Indexed {len(index.get('documents', []))} documents")
