"""Deduplicate DAPO/AIME data without changing first-occurrence row order."""
import argparse
from pathlib import Path

import pandas as pd
import polars as pl

FILES = {
    "dapo-math-17k": "dapo-math-17k.parquet",
    "aime-2024": "aime-2024.parquet",
}


def clean_dataframe(dataframe: pl.DataFrame) -> pl.DataFrame:
    # Seeded training shuffles operate on row positions, so unordered unique()
    # makes independently prepared datasets produce different prompt batches.
    deduplicated = dataframe.unique(
        subset=["data_source", "prompt", "ability", "reward_model"],
        maintain_order=True,
        keep="first",
    )
    counted = deduplicated.with_columns(pl.col("reward_model").n_unique().over("prompt").alias("n_rm"))
    return counted.filter(pl.col("n_rm") == 1).drop("n_rm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "data/dapo")
    args = parser.parse_args()
    for name, filename in FILES.items():
        dataframe = pl.from_pandas(pd.read_parquet(args.data_dir / filename))
        out_path = args.data_dir / f"{name}-cleaned.parquet"
        clean_dataframe(dataframe).to_pandas().to_parquet(out_path)
        print(f"Cleaned file saved to: {out_path}")


if __name__ == "__main__":
    main()
