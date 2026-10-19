"""Roda os baselines no M1 exportado e imprime a regua que o RL vai ter que bater."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from winrl.backtest import Costs, run  # noqa: E402
from winrl.strategies import OpeningRangeBreakout, RandomScalper  # noqa: E402

df = pd.read_parquet(ROOT / "data" / "WIN_N" / "M1.parquet")
# descarta pregoes incompletos (o primeiro do export comeca no meio, o de hoje ainda nao fechou)
counts = df.groupby(df["time"].dt.date)["time"].transform("size")
df = df[counts >= 500].reset_index(drop=True)
costs = Costs()
print(f"{df['time'].dt.date.nunique()} pregoes de M1, custo por ida e volta = {2 * costs.per_side:.1f} pts\n")

rows = []
rand = [run(df, RandomScalper(seed=s), costs).summary(f"aleatorio seed {s}") for s in range(20)]
r = pd.DataFrame(rand)
rows.append({"estrategia": "aleatorio (20 seeds, media)", "trades": int(r["trades"].mean()),
             "trades/dia": r["trades/dia"].mean(), "pts/trade": round(r["pts/trade"].mean(), 1),
             "IC95": f"pior {r['pts/trade'].min():.1f} / melhor {r['pts/trade'].max():.1f}",
             "total R$": round(r["total R$"].mean(), 2)})
rows.append(run(df, RandomScalper(seed=0), Costs(0, 0)).summary("aleatorio SEM custo"))
orb = run(df, OpeningRangeBreakout(), costs)
rows.append(orb.summary("rompimento 15min"))
rows.append(run(df, OpeningRangeBreakout(), Costs(0, 0)).summary("rompimento 15min SEM custo"))

pd.set_option("display.width", 200)
print(pd.DataFrame(rows).fillna("").to_string(index=False))
orb.frame().to_parquet(ROOT / "data" / "trades_orb.parquet", index=False)
