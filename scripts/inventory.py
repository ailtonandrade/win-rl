"""Inventario de dados do mini indice no MT5. Somente leitura: nao envia ordens.

Para cada simbolo/timeframe mostra quanto historico a corretora entrega, e com
os ticks do contrato vigente mede o spread real (ask - bid) por hora do pregao.
"""
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd

SYMBOLS = ["WIN$N", "WIN$", "WIN$D"]
TIMEFRAMES = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
TICK_DAYS = 10

if not mt5.initialize():
    raise SystemExit(f"initialize falhou: {mt5.last_error()}")
if mt5.account_info().trade_mode != 0:
    raise SystemExit("Conta nao e DEMO, abortando por seguranca.")


def rates(symbol, tf):
    # Pedir um intervalo muito longo de uma vez da "Invalid params"; vai por trimestre.
    mt5.symbol_select(symbol, True)
    parts, start = [], datetime(2010, 1, 1)
    while start < datetime.now():
        end = start + timedelta(days=91)
        r = mt5.copy_rates_range(symbol, tf, start, end)
        if r is not None and len(r):
            parts.append(pd.DataFrame(r))
        start = end
    if not parts:
        return None
    df = pd.concat(parts).drop_duplicates("time")
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


print("=== HISTORICO DE CANDLES ===")
for sym in SYMBOLS:
    for name, tf in TIMEFRAMES.items():
        df = rates(sym, tf)
        if df is None:
            print(f"{sym:6} {name:3}  sem dados ({mt5.last_error()})")
            continue
        days = df["time"].dt.date.nunique()
        rng = (df["high"] - df["low"]).median()
        print(f"{sym:6} {name:3}  {len(df):>8} candles  {df['time'].min():%Y-%m-%d} -> {df['time'].max():%Y-%m-%d %H:%M}"
              f"  {days:>5} pregoes  range mediano={rng:.0f} pts  real_vol>0={(df['real_volume'] > 0).mean():.0%}")

# Horario do pregao segundo os proprios dados (M1 do WIN$N)
m1 = rates("WIN$N", mt5.TIMEFRAME_M1)
if m1 is not None:
    last = m1[m1["time"] >= m1["time"].max() - timedelta(days=30)]
    by_day = last.groupby(last["time"].dt.date)["time"].agg(["min", "max"])
    print("\n=== SESSAO (ultimos 30 dias, horario do servidor) ===")
    print(f"primeiro candle mais comum: {by_day['min'].dt.strftime('%H:%M').mode()[0]}"
          f"   ultimo candle mais comum: {by_day['max'].dt.strftime('%H:%M').mode()[0]}")
    m1["range"] = m1["high"] - m1["low"]
    by_hour = m1[m1["time"] >= m1["time"].max() - timedelta(days=90)].groupby(m1["time"].dt.hour)["range"].median()
    print("range mediano do M1 por hora (90 dias):", ", ".join(f"{h}h={v:.0f}" for h, v in by_hour.items()))

# Contrato vigente = o vencimento futuro mais proximo
now = datetime.now().timestamp()
contracts = sorted((s for s in mt5.symbols_get("WIN*") if s.expiration_time > now), key=lambda s: s.expiration_time)
cur = contracts[0].name
mt5.symbol_select(cur, True)
print(f"\n=== TICKS ({cur}, ultimos {TICK_DAYS} dias) ===")
t = mt5.copy_ticks_range(cur, datetime.now() - timedelta(days=TICK_DAYS), datetime.now() + timedelta(days=1), mt5.COPY_TICKS_ALL)
if t is None or len(t) == 0:
    print(f"sem ticks ({mt5.last_error()})")
else:
    tk = pd.DataFrame(t)
    tk["time"] = pd.to_datetime(tk["time_msc"], unit="ms")
    print(f"{len(tk)} ticks  {tk['time'].min()} -> {tk['time'].max()}")
    q = tk[(tk["bid"] > 0) & (tk["ask"] > tk["bid"])].copy()
    q["spread"] = q["ask"] - q["bid"]
    print(f"spread: mediana={q['spread'].median():.0f} pts  media={q['spread'].mean():.1f}  p95={q['spread'].quantile(.95):.0f}"
          f"  % em 5 pts={(q['spread'] == 5).mean():.0%}")
    print("spread medio por hora:", ", ".join(f"{h}h={v:.1f}" for h, v in q.groupby(q["time"].dt.hour)["spread"].mean().items()))

    # Quanto tempo de ticks existe para tras (sonda por mes)
    oldest = None
    for months in (1, 3, 6, 12, 24, 36):
        start = datetime.now() - timedelta(days=30 * months)
        p = mt5.copy_ticks_range("WIN$N", start, start + timedelta(days=2), mt5.COPY_TICKS_ALL)
        ok = p is not None and len(p) > 0
        print(f"  ticks em WIN$N ha ~{months:>2} meses: {'sim' if ok else 'nao'}")
        if ok:
            oldest = months

mt5.shutdown()
