from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inventory(checksums: Path) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str]]]:
    code = checksums.resolve().parent
    dataset = code.parent / "dataset"
    assets, tables, seen = [], [], set()
    with checksums.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            target = (code / row["path"]).resolve()
            if target in seen:
                raise ValueError(f"Duplicate inventory path: {row['path']}")
            seen.add(target)
            if target.is_relative_to(dataset):
                relative = target.relative_to(dataset).as_posix()
                parts = PurePosixPath(relative).parts
                if len(parts) != 2 or parts[0] not in {"images", "masks"} or target.suffix != ".png":
                    raise ValueError("Only flat images/masks PNG paths are allowed")
                assets.append((target, row["sha256"], relative))
            elif target.is_relative_to(code / "bang_quy_doi"):
                tables.append((target, row["sha256"]))
            else:
                raise ValueError(f"Inventory path escapes the project data contract: {row['path']}")
    images = {relative.split("/")[1] for _, _, relative in assets if relative.startswith("images/")}
    masks = {relative.split("/")[1] for _, _, relative in assets if relative.startswith("masks/")}
    if not images or images != masks:
        raise ValueError("Inventory image/mask filenames do not match")
    return assets, tables


def write_verified(stream, target: Path, expected_hash: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".import-", delete=False) as handle:
            temporary = Path(handle.name)
            shutil.copyfileobj(stream, handle)
        if sha256(temporary) != expected_hash:
            raise ValueError(f"Imported content has the wrong SHA256: {target.name}")
        if target.exists():
            raise FileExistsError(f"Destination appeared during import; not overwriting: {target}")
        os.rename(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def initialize(checksums: Path, source_dir: Path | None = None,
               archive_path: Path | None = None) -> dict:
    if source_dir is not None and archive_path is not None:
        raise ValueError("Choose a source directory OR a ZIP, not both")
    assets, tables = read_inventory(checksums)
    for table, expected in tables:
        if not table.is_file() or sha256(table) != expected:
            raise ValueError(f"Tracked F-Z table differs from inventory: {table}")
    missing = []
    for target, expected, relative in assets:
        if target.exists():
            if not target.is_file() or sha256(target) != expected:
                raise ValueError(f"Existing data differs from inventory; refusing to overwrite: {target}")
        else:
            missing.append((target, expected, relative))
    code = checksums.resolve().parent
    if missing and source_dir is None and archive_path is None:
        candidates = [Path.home() / "reservoir8_temporal/dataset", Path.home() / "reservoir8_method/dataset"]
        source_dir = next((path for path in candidates if path.is_dir() and path.resolve() != code.parent / "dataset"), None)
        if source_dir is None:
            archives = [Path.home() / "reservoir8_3090_temporal_20261117.zip",
                        Path.home() / "reservoir8_3090_method_20261116.zip"]
            archive_path = next((path for path in archives if path.is_file()), None)
        if source_dir is None and archive_path is None:
            raise FileNotFoundError("Dataset is missing. Supply --source-dir <existing dataset folder> or --archive <previous bundle ZIP>. Transfer images/masks only once; they are intentionally not on GitHub.")
    if missing and source_dir is not None:
        source_dir = source_dir.resolve()
        if not (source_dir / "images").is_dir() and (source_dir / "dataset/images").is_dir():
            source_dir = source_dir / "dataset"
        for _, expected, relative in missing:
            source = source_dir / relative
            if not source.is_file() or sha256(source) != expected:
                raise ValueError(f"Source asset is missing or differs from inventory: {source}")
        for target, expected, relative in missing:
            with (source_dir / relative).open("rb") as stream:
                write_verified(stream, target, expected)
    elif missing and archive_path is not None:
        with ZipFile(archive_path) as archive:
            for _, _, relative in missing:
                archive.getinfo("dataset/" + relative)
            for target, expected, relative in missing:
                # Extract only allowlisted dataset assets; never code or arbitrary ZIP paths.
                with archive.open("dataset/" + relative) as stream:
                    write_verified(stream, target, expected)
    for target, expected, _ in assets:
        if sha256(target) != expected:
            raise ValueError(f"Post-import checksum mismatch: {target}")
    return {"status": "pass", "dataset_dir": str(code.parent / "dataset"),
            "image_mask_pairs": len(assets) // 2, "files_imported": len(missing),
            "existing_files_reused": len(assets) - len(missing), "fz_tables_verified": len(tables),
            "data_uploaded_to_git": False, "existing_data_overwritten": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the dataset once, independently of Git code updates")
    parser.add_argument("--checksums", type=Path, default=Path(__file__).resolve().parent / "data_checksums.csv")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    report = initialize(args.checksums, args.source_dir, args.archive)
    output = Path(__file__).resolve().parent / "runs"
    output.mkdir(exist_ok=True)
    (output / "data_import.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
