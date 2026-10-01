from __future__ import annotations

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import load_dataset, remove_duplicates


def main() -> int:
    df = load_dataset()
    cleaned, removed = remove_duplicates(df, settings.text_column)

    output = ROOT / "data/processed/dataset_clean.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)

    print(f"Filas originales: {len(df)}")
    print(f"Duplicados eliminados: {removed}")
    print(f"Filas finales: {len(cleaned)}")
    print(f"Guardado en: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())