"""Ablacao: uma base de scalp e variantes que mudam UMA coisa cada, varias seeds.

  python scripts/ablation.py                  (treina o que faltar e avalia)
  python scripts/ablation.py --eval-only

Todas as configs tem tempo maximo em posicao (scalp). Treina em paralelo (--jobs),
pula modelos ja treinados e grava reports/ablation.csv.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from evaluate import evaluate  # noqa: E402

BASE = ["--max-hold", "15"]  # features completas, rede 128x128, sem penalidade, sem tempo minimo
CONFIGS = {
    "base": [],
    "penalty": ["--penalty", "5"],
    "minhold": ["--min-hold", "5"],
    "net32": ["--net", "32", "32"],
    "small": ["--features", "small"],
}

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--steps", type=int, default=1_000_000)
p.add_argument("--jobs", type=int, default=5)
p.add_argument("--random", type=int, default=10, help="aleatorios por avaliacao")
p.add_argument("--eval-only", action="store_true")
a = p.parse_args()

py = sys.executable
runs = [(name, s) for name in CONFIGS for s in range(a.seeds)]
model_dir = lambda name, s: ROOT / "models" / f"abl-{name}_WIN_N_s{s}"  # noqa: E731

if not a.eval_only:
    (ROOT / "logs").mkdir(exist_ok=True)
    todo = [(n, s) for n, s in runs if not (model_dir(n, s) / "last_model.zip").exists()]
    print(f"treinando {len(todo)} modelos, {a.jobs} por vez")
    running, t0 = [], time.time()
    while todo or running:
        while todo and len(running) < a.jobs:
            n, s = todo.pop(0)
            cmd = [py, "-W", "ignore", "scripts/train.py", "--data", "WIN$N", "--steps", str(a.steps),
                   "--seed", str(s), "--envs", "4", "--threads", "1", "--tag", f"abl-{n}", *BASE, *CONFIGS[n]]
            log = open(ROOT / "logs" / f"abl-{n}_s{s}.log", "w")
            running.append(((n, s), subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                print(f"  {item[0][0]} s{item[0][1]} terminou (codigo {item[1].returncode}) "
                      f"{(time.time() - t0) / 60:.0f} min")
        time.sleep(5)

rows = []
for n, s in runs:
    d = model_dir(n, s)
    r = {"config": n, "seed": s}
    for split_name, f, key in (("train", "best_model.zip", "treino"), ("test", "best_model.zip", "teste best"),
                               ("test", "last_model.zip", "teste last")):
        row, _ = evaluate(d, split_name, n_random=a.random, model_file=f)
        r[f"{key} pts/trade"] = row.get("pts/trade")
        if key == "teste best":
            r.update({"trades/dia": row.get("trades/dia"), "min/trade": row.get("min/trade"),
                      "acerto": row.get("acerto"), "teste R$": row.get("total R$"), "bate aleat.": row.get("bate aleat.")})
    rows.append(r)
    print(r)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "reports" / "ablation.csv", index=False)
agg = df.groupby("config", sort=False).agg(**{
    "trades/dia": ("trades/dia", "mean"),
    "min/trade": ("min/trade", "mean"),
    "treino pts/trade": ("treino pts/trade", "mean"),
    "teste best pts/trade": ("teste best pts/trade", "mean"),
    "teste last pts/trade": ("teste last pts/trade", "mean"),
    "seeds teste>0": ("teste best pts/trade", lambda x: f"{(x > 0).sum()}/{len(x)}"),
    "teste R$ medio": ("teste R$", "mean"),
}).round(1)
pd.set_option("display.width", 250)
print("\n=== media por config (custo 9 pts/trade incluso; teste = 44 pregoes nunca usados) ===")
print(agg.to_string())
