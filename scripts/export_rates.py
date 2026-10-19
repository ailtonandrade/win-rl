"""Exporta candles do MT5 para o formato canonico em data/<SIMBOLO>/<TF>.parquet.

Somente leitura. Busca semana a semana, do presente para tras, ate achar
8 semanas seguidas sem dados (o MT5 recusa intervalos grandes demais em M1).
Uso: python export_rates.py [SIMBOLO] [TF ...]
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TFS = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}

symbol = sys.argv[1] if len(sys.argv) > 1 else "WIN$N"
wanted = sys.argv[2:] or ["M1", "M5", "M15"]

if not mt5.initialize():
    raise SystemExit(f"initialize falhou: {mt5.last_error()}")
if mt5.account_info().trade_mode != 0:
    raise SystemExit("Conta nao e DEMO, abortando por seguranca.")
mt5.symbol_select(symbol, True)

for name in wanted:
    parts, end, empty = [], datetime.now() + timedelta(days=1), 0
    while empty < 8 and end > datetime(2010, 1, 1):
        start = end - timedelta(days=7)
        r = mt5.copy_rates_range(symbol, TFS[name], start, end)
        if r is not None and len(r):
            parts.append(pd.DataFrame(r))
            empty = 0
        else:
            empty += 1
        end = start
    if not parts:
        print(f"{symbol} {name}: sem dados")
        continue
    df = pd.concat(parts).drop_duplicates("time").sort_values("time")
    # Formato canonico: horario do servidor (Clear = Brasilia), volume = contratos negociados
    out = pd.DataFrame({
        "time": pd.to_datetime(df["time"], unit="s").dt.tz_localize("America/Sao_Paulo"),
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "volume": df["real_volume"], "trades": df["tick_volume"], "spread": df["spread"],
    }).reset_index(drop=True)
    path = ROOT / "data" / symbol.replace("$", "_") / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"{symbol} {name}: {len(out)} candles, {out['time'].dt.date.nunique()} pregoes, "
          f"{out['time'].min():%Y-%m-%d} -> {out['time'].max():%Y-%m-%d}  -> {path.relative_to(ROOT)}")

mt5.shutdown()
