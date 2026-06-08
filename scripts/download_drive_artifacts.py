#!/usr/bin/env python3
"""
Скачивание артефактов из общей папки Google Drive (курс / MLOps).

Папка: https://drive.google.com/drive/folders/1HbYd0bgGCuGbBdKDmH0dCGJQt-pOHcGv

Требуется: pip install gdown
При ошибках доступа используйте браузерную авторизацию Google или dvc pull с dvc[gdrive].
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1HbYd0bgGCuGbBdKDmH0dCGJQt-pOHcGv"
)
FILES = (
    ("credit_scoring.csv", ROOT / "data" / "raw" / "credit_scoring.csv"),
    ("model_bundle_catboost.pkl", ROOT / "models" / "model_bundle_catboost.pkl"),
    ("model_bundle_xgboost.pkl", ROOT / "models" / "model_bundle_xgboost.pkl"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download data/models from Google Drive folder")
    parser.add_argument(
        "--url",
        default=DRIVE_FOLDER_URL,
        help="Google Drive folder URL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "_gdrive_download",
        help="Temporary download directory",
    )
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        print("Install gdown: pip install gdown", file=sys.stderr)
        return 1

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading folder →", out)
    gdown.download_folder(url=args.url, output=str(out), quiet=False, use_cookies=False)

    missing = []
    for name, dest in FILES:
        found = list(out.rglob(name))
        if not found:
            missing.append(name)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found[0], dest)
        print("OK", dest)

    if missing:
        print("Not found in folder:", missing, file=sys.stderr)
        return 1
    print("\nNext: dvc status  (при необходимости dvc commit после замены файлов)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
