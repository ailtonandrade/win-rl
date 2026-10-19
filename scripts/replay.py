"""Replay visual de um pregao: candles M1, volume, entradas/saidas, SL/TP e PnL acumulado.
Uso: python replay.py [--trades data/trades_orb.parquet] [--data WIN$N] [--day AAAA-MM-DD]
(padrao: o pregao com mais trades no arquivo)"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from winrl.data import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--trades", default="data/trades_orb.parquet")
p.add_argument("--data", default="WIN$N")
p.add_argument("--day")
a = p.parse_args()

bars = load(a.data)
trades = pd.read_parquet(ROOT / a.trades)
label = Path(a.trades).stem.removeprefix("trades_")

day = pd.to_datetime(a.day).date() if a.day else trades["day"].value_counts().idxmax()
d = bars[bars["time"].dt.date == day]
tr = trades[trades["day"] == day]

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.18, 0.2], vertical_spacing=0.03,
                    subplot_titles=(f"{a.data} M1 - {day}  ({label}, custos inclusos)", "volume (contratos)", "PnL acumulado (pts)"))
fig.add_trace(go.Candlestick(x=d["time"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                             name="WIN$N", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), 1, 1)
fig.add_trace(go.Bar(x=d["time"], y=d["volume"], marker_color="#90a4ae", name="volume"), 2, 1)

# faixa de abertura usada pela estrategia
first = d.iloc[:15]
fig.add_hrect(y0=first["low"].min(), y1=first["high"].max(), fillcolor="#ffd54f", opacity=0.12, line_width=0, row=1, col=1)

pnl = pd.Series(0.0, index=d["time"])
for _, t in tr.iterrows():
    long = t["side"] > 0
    fig.add_trace(go.Scatter(x=[t["entry_time"]], y=[t["entry"]], mode="markers+text",
                             marker=dict(symbol="triangle-up" if long else "triangle-down", size=16,
                                         color="#1e88e5" if long else "#e53935"),
                             text=["COMPRA" if long else "VENDA"], textposition="bottom center", name="entrada"), 1, 1)
    fig.add_trace(go.Scatter(x=[t["exit_time"]], y=[t["exit"]], mode="markers+text",
                             marker=dict(symbol="x", size=14, color="#212121"),
                             text=[f"{t['pts']:+.0f}" if len(tr) > 6 else f"{t['reason']} {t['pts']:+.0f} pts"],
                             textposition="top center", name="saida"), 1, 1)
    for level, color, tag in ((t["sl"], "#e53935", "SL"), (t["tp"], "#43a047", "TP")):
        if pd.notna(level):
            fig.add_shape(type="line", x0=t["entry_time"], x1=t["exit_time"], y0=level, y1=level,
                          line=dict(color=color, dash="dash"), row=1, col=1)
            fig.add_annotation(x=t["entry_time"], y=level, text=tag, showarrow=False, xanchor="right",
                               font=dict(color=color), row=1, col=1)
    pnl[pnl.index >= t["exit_time"]] += t["pts"]
fig.add_trace(go.Scatter(x=pnl.index, y=pnl.values, line=dict(color="#5e35b1"), fill="tozeroy", name="PnL"), 3, 1)

fig.update_layout(height=820, showlegend=False, xaxis_rangeslider_visible=False, template="plotly_white",
                  margin=dict(l=50, r=20, t=50, b=30))
out = ROOT / "reports" / f"replay_{label}_{day}.html"
out.parent.mkdir(exist_ok=True)
fig.write_html(out, include_plotlyjs="cdn")
print(out)
