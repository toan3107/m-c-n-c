from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from initialize_data import initialize, read_inventory


class DataInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "project/code"
        self.code.mkdir(parents=True)
        self.source = self.root / "source"
        records = []
        for folder, content in (("images", b"image-content"), ("masks", b"mask-content")):
            source = self.source / folder / "fixture__2020-01-01.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(content)
            records.append({"path": "../dataset/" + folder + "/" + source.name,
                            "sha256": hashlib.sha256(content).hexdigest()})
        self.checksums = self.code / "data_checksums.csv"
        self.write_records(records)

    def write_records(self, records):
        with self.checksums.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "sha256"])
            writer.writeheader()
            writer.writerows(records)

    def test_import_directory_is_idempotent(self):
        first = initialize(self.checksums, source_dir=self.source)
        second = initialize(self.checksums)
        self.assertEqual(first["files_imported"], 2)
        self.assertEqual(second["files_imported"], 0)
        self.assertEqual(second["image_mask_pairs"], 1)

    def test_previous_bundle_zip_extracts_only_dataset_members(self):
        archive = self.root / "previous.zip"
        with ZipFile(archive, "w") as handle:
            for file in self.source.rglob("*.png"):
                handle.write(file, "dataset/" + file.relative_to(self.source).as_posix())
            handle.writestr("code/train.py", "untrusted old code")
            handle.writestr("../escape.txt", "not extracted")
        report = initialize(self.checksums, archive_path=archive)
        self.assertEqual(report["files_imported"], 2)
        self.assertFalse((self.code / "train.py").exists())
        self.assertFalse((self.root / "escape.txt").exists())

    def test_existing_modified_mask_is_never_overwritten(self):
        initialize(self.checksums, source_dir=self.source)
        mask = self.code.parent / "dataset/masks/fixture__2020-01-01.png"
        mask.write_bytes(b"manual correction")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            initialize(self.checksums, source_dir=self.source)
        self.assertEqual(mask.read_bytes(), b"manual correction")

    def test_wrong_source_is_rejected_before_import(self):
        (self.source / "images/fixture__2020-01-01.png").write_bytes(b"wrong content")
        with self.assertRaisesRegex(ValueError, "Source asset"):
            initialize(self.checksums, source_dir=self.source)
        self.assertFalse((self.code.parent / "dataset").exists())

    def test_wrong_archive_content_does_not_create_destination_file(self):
        archive = self.root / "wrong.zip"
        with ZipFile(archive, "w") as handle:
            for file in self.source.rglob("*.png"):
                handle.writestr("dataset/" + file.relative_to(self.source).as_posix(), b"wrong content")
        with self.assertRaisesRegex(ValueError, "wrong SHA256"):
            initialize(self.checksums, archive_path=archive)
        self.assertEqual(list((self.code.parent / "dataset").rglob("*.png")), [])
        self.assertEqual(list((self.code.parent / "dataset").rglob(".import-*")), [])

    def test_escaping_inventory_path_is_rejected(self):
        self.write_records([{"path": "../../outside.png", "sha256": "0" * 64}])
        with self.assertRaisesRegex(ValueError, "escapes"):
            read_inventory(self.checksums)


if __name__ == "__main__":
    unittest.main()
