"""Simulador de day trade/scalp no WIN com custos, em pontos.

Regras (as mesmas que o ambiente de RL vai usar depois):
- a estrategia decide no FECHAMENTO do candle i e a ordem executa na ABERTURA do i+1;
- 1 contrato, posicao em {-1, 0, +1}; nunca dorme posicionado: zera no ultimo candle do dia;
- custo por lado = meio spread + taxas (em pontos), cobrado na entrada e na saida;
- SL/TP sao checados dentro do candle pela maxima/minima; se os dois cabem no mesmo
  candle, assume-se o STOP (pessimista, porque o M1 nao diz qual veio primeiro).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

POINT_VALUE_BRL = 0.20  # R$ por ponto por contrato


@dataclass
class Costs:
    half_spread: float = 3.25  # spread medio medido nos ticks ~6.5 pts
    fees: float = 1.25         # emolumentos/taxas B3 por lado, aproximado em pontos
    slippage: float = 0.0

    @property
    def per_side(self):
        return self.half_spread + self.fees + self.slippage


@dataclass
class Trade:
    day: object
    side: int
    entry_time: pd.Timestamp
    entry: float
    sl: float | None
    tp: float | None
    exit_time: pd.Timestamp | None = None
    exit: float | None = None
    reason: str = ""
    pts: float = 0.0  # liquido de custos


@dataclass
class Order:
    """O que a estrategia pede ao fechar um candle."""
    side: int                  # +1 compra, -1 venda, 0 zerar
    sl_pts: float | None = None
    tp_pts: float | None = None


@dataclass
class Result:
    trades: list = field(default_factory=list)

    def frame(self):
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def summary(self, label=""):
        df = self.frame()
        if df.empty:
            return {"estrategia": label, "trades": 0}
        pts = df["pts"].to_numpy()
        se = pts.std(ddof=1) / np.sqrt(len(pts)) if len(pts) > 1 else np.nan
        equity = np.cumsum(pts)
        dd = (np.maximum.accumulate(equity) - equity).max()
        return {
            "estrategia": label,
            "trades": len(pts),
            "trades/dia": round(len(pts) / df["day"].nunique(), 1),
            "acerto": f"{(pts > 0).mean():.0%}",
            "pts/trade": round(pts.mean(), 1),
            "IC95": f"[{pts.mean() - 1.96 * se:.1f}, {pts.mean() + 1.96 * se:.1f}]",
            "total pts": round(pts.sum()),
            "total R$": round(pts.sum() * POINT_VALUE_BRL, 2),
            "max DD pts": round(dd),
        }


def run(df: pd.DataFrame, strategy, costs: Costs = Costs(), last_entry_minutes=30) -> Result:
    """strategy(day_df, i, position) -> Order | None, chamada no fechamento de cada candle."""
    res = Result()
    for day, d in df.groupby(df["time"].dt.date, sort=True):
        d = d.reset_index(drop=True)
        t = d["time"]
        o, h, l, c =(d[k].to_numpy() for k in ("open", "high", "low", "close"))
        no_entry_after = t.iloc[-1] - pd.Timedelta(minutes=last_entry_minutes)
        pos, cur, pending = 0, None, None

        def close(i, price, reason, at_open=False):
            nonlocal pos, cur
            cur.exit_time = t.iloc[i]
            cur.exit = price
            cur.reason = reason
            cur.pts = cur.side * (price - cur.entry) - 2 * costs.per_side
            res.trades.append(cur)
            pos, cur = 0, None

        if hasattr(strategy, "reset"):
            strategy.reset(day)

        for i in range(len(d)):
            # 1) executa a ordem decidida no fechamento anterior, na abertura deste candle
            if pending is not None:
                if pos != 0 and pending.side != pos:
                    close(i, o[i], "sinal")
                if pending.side != 0 and pos == 0 and t.iloc[i] <= no_entry_after:
                    sl = o[i] - pending.side * pending.sl_pts if pending.sl_pts else None
                    tp = o[i] + pending.side * pending.tp_pts if pending.tp_pts else None
                    cur = Trade(day, pending.side, t.iloc[i], o[i], sl, tp)
                    pos = pending.side
                pending = None

            # 2) SL/TP dentro do candle (pessimista se os dois couberem)
            if pos != 0:
                hit_sl = cur.sl is not None and (l[i] <= cur.sl if pos > 0 else h[i] >= cur.sl)
                hit_tp = cur.tp is not None and (h[i] >= cur.tp if pos > 0 else l[i] <= cur.tp)
                if hit_sl:
                    # gap alem do stop executa no pior preco disponivel (a abertura)
                    px = min(cur.sl, o[i]) if pos > 0 else max(cur.sl, o[i])
                    close(i, px, "stop")
                elif hit_tp:
                    close(i, cur.tp, "alvo")

            # 3) ultimo candle do dia: zera no fechamento
            if i == len(d) - 1:
                if pos != 0:
                    close(i, c[i], "fim do dia")
                break

            # 4) estrategia decide no fechamento
            order = strategy(d, i, pos)
            if order is not None and (order.side != pos):
                pending = order
    return res
