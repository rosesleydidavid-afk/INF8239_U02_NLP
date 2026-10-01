from __future__ import annotations

import io
import sys
import zipfile

import pandas as pd
import requests

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import sha256

SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def main() -> int:
    if settings.data_source != "local":
        print("Este proyecto descarga el SMS Spam Collection directamente del ZIP oficial de UCI.", file=sys.stderr)
        print("Configure DATA_SOURCE=local y DATASET_PATH=data/raw/dataset.csv en .env", file=sys.stderr)
        return 2

    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open("SMSSpamCollection") as file:
            frame = pd.read_csv(
                file, sep="\t", header=None, names=["label", "text"], encoding="latin-1"
            )

    output = ROOT / "data/raw/dataset.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print(f"Guardado: {output}")
    print(f"Filas: {len(frame)}")
    print(f"SHA-256: {sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
