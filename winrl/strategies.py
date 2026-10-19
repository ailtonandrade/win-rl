"""Baselines. O agente de RL so "aprendeu algo" se bater estes fora da amostra."""
import numpy as np

from .backtest import Order


class RandomScalper:
    """Entra aleatoriamente comprado ou vendido, com o mesmo SL/TP que um scalp usaria.
    E a regua minima: quanto se perde operando sem nenhuma informacao."""

    def __init__(self, seed=0, entries_per_day=6, sl=60, tp=60, bars_per_day=540):
        self.rng = np.random.default_rng(seed)
        self.p = entries_per_day / bars_per_day
        self.sl, self.tp = sl, tp

    def __call__(self, d, i, pos):
        if pos == 0 and self.rng.random() < self.p:
            return Order(int(self.rng.choice([-1, 1])), self.sl, self.tp)
        return None


class OpeningRangeBreakout:
    """Rompimento da faixa dos primeiros N minutos, 1 trade por dia.
    Stop no outro lado da faixa (limitado), alvo = rr x risco."""

    def __init__(self, minutes=15, rr=1.5, max_risk=250, min_risk=40):
        self.minutes, self.rr, self.max_risk, self.min_risk = minutes, rr, max_risk, min_risk

    def reset(self, day):
        self.done = False

    def __call__(self, d, i, pos):
        if self.done or pos != 0 or i < self.minutes:
            return None
        hi = d["high"].iloc[: self.minutes].max()
        lo = d["low"].iloc[: self.minutes].min()
        risk = float(np.clip(hi - lo, self.min_risk, self.max_risk))
        close = d["close"].iloc[i]
        if close > hi or close < lo:
            self.done = True
            return Order(1 if close > hi else -1, risk, risk * self.rr)
        return None
