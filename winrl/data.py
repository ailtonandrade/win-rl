"""Carrega dados no formato canonico e separa treino / validacao / teste por DATA."""
from pathlib import Path

import pandas as pd

from .synthetic import make_days

ROOT = Path(__file__).resolve().parents[1]


def load(source: str) -> pd.DataFrame:
    if source in ("random", "momentum"):
        return make_days(240, kind=source, seed=42)
    df = pd.read_parquet(ROOT / "data" / source.replace("$", "_") / "M1.parquet")
    # so pregoes completos (o primeiro do export comeca no meio, o de hoje pode estar aberto)
    size = df.groupby(df["time"].dt.date)["time"].transform("size")
    return df[size >= 500].reset_index(drop=True)


def split(df: pd.DataFrame, train=0.6, val=0.15):
    days = sorted(df["time"].dt.date.unique())
    a, b = int(len(days) * train), int(len(days) * (train + val))
    pick = lambda ds: df[df["time"].dt.date.isin(set(ds))].reset_index(drop=True)  # noqa: E731
    return pick(days[:a]), pick(days[a:b]), pick(days[b:])
