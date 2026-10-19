"""Pregoes sinteticos no formato canonico, para testar se o pipeline aprende.

- "random":   passeio aleatorio puro. Nao ha nada a aprender: o agente certo NAO opera.
- "momentum": retornos autocorrelacionados (phi). Ha vantagem real acima do custo:
              o agente certo TEM que lucrar fora da amostra.
"""
import numpy as np
import pandas as pd


def make_days(n_days: int, kind="random", phi=0.4, seed=0, start="2025-01-02", bars=565) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    minute = np.arange(bars)
    # volatilidade em "U": abertura e fechamento mais agitados
    sigma = 22 + 25 * np.exp(-minute / 30) + 8 * np.exp(-(bars - minute) / 40)
    out, price = [], 130_000.0
    for d in days:
        eps = rng.standard_normal(bars) * sigma
        ret = np.empty(bars)
        prev = 0.0
        for k in range(bars):
            prev = (phi * prev if kind == "momentum" else 0.0) + eps[k]
            ret[k] = prev
        price += rng.normal(0, 150)  # gap de abertura
        close = price + np.cumsum(ret)
        open_ = np.r_[price, close[:-1]]
        wick = np.abs(rng.normal(0, sigma * 0.5, (2, bars)))
        high = np.maximum(open_, close) + wick[0]
        low = np.minimum(open_, close) - wick[1]
        rnd = lambda x: np.round(x / 5) * 5  # noqa: E731  tick de 5 pontos
        t = pd.Timestamp(d).tz_localize("America/Sao_Paulo") + pd.Timedelta(hours=9) + pd.to_timedelta(minute, "min")
        out.append(pd.DataFrame({
            "time": t, "open": rnd(open_), "high": rnd(high), "low": rnd(low), "close": rnd(close),
            "volume": rng.lognormal(10, 0.5, bars).astype(int), "trades": 0, "spread": 5,
        }))
        price = close[-1]
    return pd.concat(out, ignore_index=True)
