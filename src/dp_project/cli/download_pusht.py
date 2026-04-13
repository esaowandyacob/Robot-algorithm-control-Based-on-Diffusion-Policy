from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract the official Push-T dataset.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dataset zip URL.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory. The dataset will be extracted under this path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract even if files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "pusht.zip"
    extracted_root = data_dir / "pusht"
    dataset_path = extracted_root / "pusht_cchi_v7_replay.zarr"

    if args.force or not archive_path.exists():
        print(f"Downloading Push-T archive to {archive_path}")
        urlretrieve(args.url, archive_path)
    else:
        print(f"Archive already exists: {archive_path}")

    if args.force and extracted_root.exists():
        for path in sorted(extracted_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        extracted_root.rmdir()

    if not dataset_path.exists():
        print(f"Extracting archive into {data_dir}")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_dir)
    else:
        print(f"Extracted dataset already exists: {dataset_path}")

    print(f"Push-T dataset ready at: {dataset_path}")

