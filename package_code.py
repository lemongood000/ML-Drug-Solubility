from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
DIST_DIR = ROOT / "dist"
ARCHIVE = DIST_DIR / "COMP6133_ML_Code_Final.zip"
EXCLUDED_PARTS = {
    ".venv",
    ".venv-clean",
    ".git",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    "dist",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".log",
}


def include_path(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if relative.parts[0] == "report_assets" and path.suffix.lower() == ".docx":
        return False
    return path.is_file()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in ROOT.rglob("*") if include_path(path))
    dataset_path = PARENT / "data_2026.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    with zipfile.ZipFile(
        ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in files:
            archive.write(
                path,
        arcname=Path("COMP6133_ML_Code_Final") / "代码" / path.relative_to(ROOT),
            )
        archive.write(
            dataset_path,
            arcname=Path("COMP6133_ML_Code_Final") / "data_2026.csv",
        )
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "source_file_count": len(files) + 1,
            "excluded": sorted(EXCLUDED_PARTS),
            "archive_sha256_pending": True,
        }
        archive.writestr(
            str(Path("COMP6133_ML_Code_Final") / "PACKAGE_MANIFEST.json"),
            json.dumps(manifest, indent=2),
        )

    archive_hash = sha256(ARCHIVE)
    (DIST_DIR / "COMP6133_ML_Code_Final.sha256").write_text(
        f"{archive_hash}  {ARCHIVE.name}\n", encoding="utf-8"
    )
    print(f"Archive: {ARCHIVE}")
    print(f"Size: {ARCHIVE.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"SHA256: {archive_hash}")


if __name__ == "__main__":
    main()
