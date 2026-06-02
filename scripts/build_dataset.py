"""Fetch the dataset from OpenML, cache it, and write the committed sample.

Run:  python scripts/build_dataset.py
"""

import logging

from frauddet.data.load import build_sample, load_creditcard

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    df = load_creditcard()
    fraud = int(df["Class"].sum())
    print(f"\n✓ loaded {len(df):,} transactions — {fraud} fraud ({df['Class'].mean() * 100:.3f}%)")
    print("  columns:", list(df.columns[:5]), "…", list(df.columns[-3:]))
    path = build_sample()
    print(f"✓ committed sample → {path}")


if __name__ == "__main__":
    main()
