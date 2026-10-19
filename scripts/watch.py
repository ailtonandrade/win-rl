"""Modo ASSISTIR: um modelo ja treinado (congelado) joga os pregoes e publica no painel.

  python scripts/watch.py --model models/abl-penalty_WIN_N_s2 --split test
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from winrl.data import load, split  # noqa: E402
from winrl.env import EnvConfig, WinDayTradeEnv  # noqa: E402
from winrl.live import Board, record_episode  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="pasta do modelo")
p.add_argument("--file", default="best_model.zip")
p.add_argument("--split", default="test", choices=["train", "val", "test"])
a = p.parse_args()

d = ROOT / a.model if not Path(a.model).is_absolute() else Path(a.model)
meta = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {"data": "WIN$N", "env": {}}
df = dict(zip(("train", "val", "test"), split(load(meta["data"]))))[a.split]
cfg = EnvConfig(**meta["env"])
env = WinDayTradeEnv(df, cfg, random_days=False)
model = PPO.load(d / a.file, device="cpu")

board = Board("assistir", f"{d.name} / {a.file} / {a.split}", {**cfg.to_dict(), **{k: meta.get(k) for k in ("net", "steps", "seed")}},
              keep=1000)
for k in range(len(env.days)):
    ep = record_episode(env, model, k)
    board.add_episode(ep)
    print(f"{ep['day']}: {ep['summary']['trades']} trades, {ep['summary']['pnl_pts']:+.0f} pts")
print("pronto - todos os pregoes publicados no painel")
