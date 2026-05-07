import hashlib
import json
import os
import re
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

LAW_ASSETS_DIR = Path(os.getenv("LAW_ASSETS_DIR", "law_assets/library"))
LAW_ASSETS_INDEX = Path(os.getenv("LAW_ASSETS_INDEX", "law_assets/index/documents.json"))
LAW_ASSETS_SOURCE_DIR = os.getenv("LAW_ASSETS_SOURCE_DIR", "")

if not LAW_ASSETS_DIR.is_absolute():
    LAW_ASSETS_DIR = PROJECT_ROOT / LAW_ASSETS_DIR
if not LAW_ASSETS_INDEX.is_absolute():
    LAW_ASSETS_INDEX = PROJECT_ROOT / LAW_ASSETS_INDEX

CATEGORY_CONFIG = {
    "tax": {
        "zip": "tax.zip",
        "source_dir": "tax",
        "name": "Tax Law",
        "icon": "TAX",
        "description": "Taxation statutes and guidance",
    },
    "banking": {
        "zip": "Banking.zip",
        "source_dir": "Banking",
        "name": "Banking Law",
        "icon": "BANK",
        "description": "Banking operations and compliance",
    },
    "cdc": {
        "zip": "CouncilForDevelopmentOfCambodia.zip",
        "source_dir": "CouncilForDevelopmentOfCambodia",
        "name": "CDC",
        "icon": "CDC",
        "description": "Council for Development of Cambodia",
    },
    "finance": {
        "zip": "Finance.zip",
        "source_dir": "Finance",
        "name": "Finance Law",
        "icon": "FIN",
        "description": "Finance and securities regulations",
    },
    "labour": {
        "zip": "labour.zip",
        "source_dir": "labour",
        "name": "Labour Law",
        "icon": "LAB",
        "description": "Employment and worker protections",
    },
    "business-registration": {
        "zip": "RegistrationBusiness.zip",
        "source_dir": "RegistrationBusiness",
        "name": "Business Registration",
        "icon": "BUS",
        "description": "Company formation and licenses",
    },
    "law-documents": {
        "zip": "LawDocuments.zip",
        "source_dir": "LawDocuments",
        "name": "Law Documents",
        "icon": "LAW",
        "description": "General Cambodian legal documents",
    },
}

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json"}
DISPLAY_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".doc", ".docx"}


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "document"


def _title(path: Path) -> str:
    return " ".join(path.stem.replace("_", " ").replace("-", " ").split())[:120] or path.name


def _size_label(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _year_from_name(path: Path) -> str:
    match = re.search(r"(19|20)\d{2}", path.name)
    return match.group(0) if match else ""


def _read_text(path: Path, max_chars: int | None = None) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    return text[:max_chars] if max_chars else text


def _short_asset_name(original_name: str, relative_name: str) -> str:
    source = Path(original_name)
    digest = hashlib.sha1(relative_name.encode("utf-8", errors="ignore")).hexdigest()[:10]
    base = _slug(source.stem)[:90] or "document"
    return f"{base}-{digest}{source.suffix.lower()}"


def _copy_material_dir(source_dir: Path, target: Path) -> None:
    if target.exists():
        _remove_tree(target)
    target.mkdir(parents=True, exist_ok=True)

    for source_path in source_dir.rglob("*"):
        if not source_path.is_file() or source_path.suffix.lower() not in DISPLAY_EXTENSIONS:
            continue
        rel = source_path.relative_to(source_dir).as_posix()
        destination = target / _short_asset_name(source_path.name, rel)
        shutil.copy2(source_path, destination)


def _extract_material_zip(source_zip: Path, target: Path) -> None:
    if target.exists():
        _remove_tree(target)
    target.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_zip, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.suffix.lower() not in DISPLAY_EXTENSIONS:
                continue
            destination = target / _short_asset_name(member_path.name, member.filename)
            with archive.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def _remove_tree(path: Path) -> None:
    def _on_error(func, target, _exc_info):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onerror=_on_error)


def prepare_assets_from_source() -> dict[str, Any]:
    if not LAW_ASSETS_SOURCE_DIR:
        raise FileNotFoundError("LAW_ASSETS_SOURCE_DIR is not set.")
    source_root = Path(LAW_ASSETS_SOURCE_DIR)
    if not source_root.exists():
        raise FileNotFoundError(f"LAW_ASSETS_SOURCE_DIR does not exist: {source_root}")

    LAW_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    prepared = []
    for category_id, config in CATEGORY_CONFIG.items():
        target = LAW_ASSETS_DIR / category_id
        marker = target / ".prepared"
        if target.exists() and marker.exists():
            prepared.append(category_id)
            continue

        source_dir = source_root / config["source_dir"]
        source_zip = source_root / config["zip"]
        if source_dir.exists():
            _copy_material_dir(source_dir, target)
            marker.write_text("ok", encoding="utf-8")
            prepared.append(category_id)
        elif source_zip.exists():
            _extract_material_zip(source_zip, target)
            marker.write_text("ok", encoding="utf-8")
            prepared.append(category_id)

    return {"prepared": prepared, "assetsDir": str(LAW_ASSETS_DIR)}


def build_law_index() -> dict[str, Any]:
    if not LAW_ASSETS_DIR.exists() or not any(LAW_ASSETS_DIR.iterdir()):
        prepare_assets_from_source()

    documents = []
    for category_id, config in CATEGORY_CONFIG.items():
        category_dir = LAW_ASSETS_DIR / category_id
        if not category_dir.exists():
            continue

        for path in sorted(p for p in category_dir.rglob("*") if p.is_file()):
            if path.suffix.lower() not in DISPLAY_EXTENSIONS:
                continue
            rel = path.relative_to(category_dir).as_posix()
            doc_id = f"{category_id}/{_slug(rel)}"
            stat = path.stat()
            documents.append(
                {
                    "id": doc_id,
                    "categoryId": category_id,
                    "title": _title(path),
                    "subtitle": config["description"],
                    "year": _year_from_name(path),
                    "pages": None,
                    "size": _size_label(stat.st_size),
                    "sourcePath": f"{category_id}/{rel}",
                    "relativePath": rel,
                    "extension": path.suffix.lower(),
                }
            )

    index = {
        "generatedFrom": "LAW_ASSETS_DIR",
        "documents": documents,
    }
    LAW_ASSETS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    LAW_ASSETS_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def load_law_index() -> dict[str, Any]:
    if not LAW_ASSETS_INDEX.exists():
        return build_law_index()
    return json.loads(LAW_ASSETS_INDEX.read_text(encoding="utf-8"))


def list_categories() -> list[dict[str, Any]]:
    index = load_law_index()
    counts: dict[str, int] = {}
    for doc in index.get("documents", []):
        counts[doc["categoryId"]] = counts.get(doc["categoryId"], 0) + 1

    categories = []
    for category_id, config in CATEGORY_CONFIG.items():
        categories.append(
            {
                "id": category_id,
                "icon": config["icon"],
                "name": config["name"],
                "description": config["description"],
                "documentCount": counts.get(category_id, 0),
            }
        )
    return categories


def list_documents(category_id: str) -> list[dict[str, Any]]:
    index = load_law_index()
    docs = [
        _summary(doc)
        for doc in index.get("documents", [])
        if doc.get("categoryId") == category_id
    ]
    return sorted(docs, key=lambda item: item["title"])


def normalize_document_id(document_id: str) -> str:
    return unquote(document_id or "").strip()


def find_document(document_id: str) -> dict[str, Any] | None:
    normalized_id = normalize_document_id(document_id)
    index = load_law_index()
    return next((item for item in index.get("documents", []) if item.get("id") == normalized_id), None)


def find_document_for_source(category_id: str, source_file: str = "", doc_id: str = "") -> dict[str, Any] | None:
    normalized_doc_id = normalize_document_id(doc_id)
    exact = find_document(normalized_doc_id)
    if exact:
        return exact

    index = load_law_index()
    docs = [
        doc for doc in index.get("documents", [])
        if not category_id or doc.get("categoryId") == category_id
    ]
    if not docs:
        return None

    candidates = [_slug(value) for value in (source_file, doc_id, Path(source_file).stem, Path(doc_id).stem) if value]
    candidates = [value for value in candidates if value]
    if not candidates:
        return None

    for doc in docs:
        searchable = " ".join(
            _slug(str(doc.get(key, "")))
            for key in ("id", "title", "relativePath", "sourcePath")
        )
        if any(candidate and (candidate in searchable or searchable in candidate) for candidate in candidates):
            return doc

    return None


def get_document(document_id: str) -> dict[str, Any] | None:
    doc = find_document(document_id)
    if not doc:
        return None
    path = LAW_ASSETS_DIR / doc["sourcePath"]
    content = ""
    if path.suffix.lower() in TEXT_EXTENSIONS and path.exists():
        content = _read_text(path)
    elif path.exists():
        content = f"This document is available for download: {path.name}"

    detail = _summary(doc)
    detail["content"] = content
    return detail


def get_document_path(document_id: str) -> Path | None:
    doc = find_document(document_id)
    if not doc:
        return None
    path = LAW_ASSETS_DIR / doc["sourcePath"]
    return path if path.exists() else None


def _summary(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doc["id"],
        "categoryId": doc["categoryId"],
        "title": doc["title"],
        "subtitle": doc["subtitle"],
        "year": doc.get("year") or "Unknown",
        "pages": doc.get("pages"),
        "size": doc.get("size"),
    }
