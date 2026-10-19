"""Modo AVALIACAO: pesos congelados, dados que o modelo nunca viu, contra os baselines.

  python scripts/evaluate.py models/WIN_N_s0                       (um modelo)
  python scripts/evaluate.py "models/small_*" --split test          (varias seeds, com resumo)

Cada pasta de modelo traz um config.json (gravado pelo train.py) com o ambiente exato
do treino; sem ele assume o ambiente padrao (modelos antigos).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from winrl.data import load, split  # noqa: E402
from winrl.env import EnvConfig, WinDayTradeEnv  # noqa: E402


class RandomHold:
    """Aleatorio com a MESMA frequencia de entradas e duracao media do agente."""

    def __init__(self, seed, p_enter, p_exit):
        self.rng, self.p_enter, self.p_exit = np.random.default_rng(seed), p_enter, p_exit

    def __call__(self, obs):
        pos = obs[-3]
        if pos == 0:
            return int(self.rng.choice([1, 2])) if self.rng.random() < self.p_enter else 0
        return 0 if self.rng.random() < self.p_exit else (1 if pos > 0 else 2)


_data_cache = {}


def evaluate(model_dir: Path, which: str, n_random=30, model_file="best_model.zip"):
    cfg_path = model_dir / "config.json"
    meta = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"data": "WIN$N", "env": {}}
    data = meta["data"]
    if data not in _data_cache:
        _data_cache[data] = dict(zip(("train", "val", "test"), split(load(data))))
    df = _data_cache[data][which]
    cfg = EnvConfig(**meta["env"])
    env = WinDayTradeEnv(df, cfg, random_days=False)
    model = PPO.load(model_dir / model_file, device="cpu")

    agent = env.run_policy(lambda o: model.predict(o, deterministic=True)[0])
    row = agent.summary(model_dir.name)
    tf = agent.frame()
    if len(tf):
        hold = max(((tf["exit_time"] - tf["entry_time"]).dt.total_seconds() / 60).mean(), 1)
        # chance de entrar por candle ZERADO (nao por candle total), para igualar o numero de trades
        flat_bars = max(len(df) - len(tf) * hold, len(tf))
        rnd = pd.DataFrame([env.run_policy(RandomHold(s, min(len(tf) / flat_bars, 1.0), 1 / hold)).summary()
                            for s in range(n_random)])
        row["aleat. pts/trade"] = round(rnd["pts/trade"].mean(), 1)
        row["bate aleat."] = f"{(rnd['pts/trade'] < row['pts/trade']).mean():.0%}"
        row["min/trade"] = round(hold, 1)
        tf.to_parquet(ROOT / "data" / f"trades_{model_dir.name}_{which}.parquet", index=False)
    return row, df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="pastas de modelo (aceita curinga)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--file", default="best_model.zip")
    a = p.parse_args()

    dirs = sorted({Path(d) for pat in a.models for d in glob.glob(pat) if Path(d).is_dir()})
    rows = []
    for d in dirs:
        row, df = evaluate(d, a.split, model_file=a.file)
        rows.append(row)
    pd.set_option("display.width", 250)
    print(f"{a.split}: {df['time'].dt.date.nunique()} pregoes {df['time'].min():%Y-%m-%d} -> {df['time'].max():%Y-%m-%d}"
          "   (custo 9 pts por ida e volta incluso)")
    cols = ["estrategia", "trades", "trades/dia", "min/trade", "acerto", "pts/trade", "IC95",
            "total R$", "max DD pts", "aleat. pts/trade", "bate aleat."]
    out = pd.DataFrame(rows).reindex(columns=cols)
    print(out.fillna("").to_string(index=False))
    if len(rows) > 1:
        r = out.dropna(subset=["pts/trade"])
        print(f"\n{len(r)} seeds: pts/trade medio {r['pts/trade'].mean():.1f} (min {r['pts/trade'].min():.1f}, "
              f"max {r['pts/trade'].max():.1f}) | seeds com pts/trade > 0: {(r['pts/trade'] > 0).sum()} | "
              f"total R$ somado {r['total R$'].sum():.2f}")
