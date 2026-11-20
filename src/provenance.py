from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


PIPELINE_VERSION = "1.1.0"
TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "joblib",
    "scikit-learn",
    "rdkit",
    "torch",
    "xgboost",
    "streamlit",
    "matplotlib",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_config_id(config: dict[str, Any], length: int = 12) -> str:
    serialized = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def package_versions() -> dict[str, str]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config_id": stable_config_id(config),
        "config": config,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
        },
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
