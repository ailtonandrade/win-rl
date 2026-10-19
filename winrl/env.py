"""Ambiente Gymnasium de day trade/scalp no WIN, 1 episodio = 1 pregao.

Mesmas regras de execucao do backtest.py:
- o agente decide no fechamento do candle i, a ordem executa na abertura do i+1;
- acoes: 0 = zerado, 1 = comprado, 2 = vendido (1 contrato);
- custo por lado (meio spread + taxas) a cada mudanca de posicao; virar a mao paga 2 lados;
- sem novas entradas nos ultimos 30 min; zera obrigatoriamente no ultimo candle.

Reward = PnL marcado a mercado do passo, em pontos, liquido de custos (x reward_scale),
menos uma penalidade opcional por entrada. Sem penalidade, a soma dos rewards de um
episodio e exatamente o resultado do dia; info["pnl_pts"] e sempre o PnL real.

Contra overtrading (EnvConfig):
- trade_penalty: pontos a mais cobrados SO no reward a cada entrada (nao entra no PnL);
- min_hold: candles minimos em posicao antes de poder sair ou virar (o fim do dia ainda zera);
- max_hold: candles maximos em posicao; ao atingir, zera na abertura seguinte ("tempo max").
"""
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import pandas as pd

from .backtest import Costs, Result, Trade
from .features import CLIP, market_features, n_features, position_features

ACTION_TO_POS = (0, 1, -1)


@dataclass
class EnvConfig:
    feature_set: str = "full"
    trade_penalty: float = 0.0
    min_hold: int = 0
    max_hold: int = 0  # 0 = sem limite; >0 = zera sozinho apos N candles (mantem o scalp curto)

    def to_dict(self):
        return asdict(self)


class WinDayTradeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, config: EnvConfig = EnvConfig(), costs: Costs = Costs(),
                 reward_scale=0.01, random_days=True, last_entry_minutes=30, seed=None,
                 session_close: str | None = None):
        super().__init__()
        self.config = config
        df = df.sort_values("time").reset_index(drop=True)
        self.feats = market_features(df, config.feature_set, session_close).to_numpy()
        self.o, self.h, self.l, self.c = (df[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
        self.atr = (df["high"] - df["low"]).rolling(20, min_periods=1).mean().clip(lower=5.0).to_numpy()
        self.time = df["time"]
        days = df.groupby(df["time"].dt.date).indices
        self.days = [(d, idx[0], idx[-1]) for d, idx in sorted(days.items())]
        if session_close:  # tempo real: o fim do dia ainda nao existe nos dados
            close = df["time"].dt.normalize() + pd.Timedelta(session_close + ":00")
        else:
            close = df.groupby(df["time"].dt.date)["time"].transform("max")
        cutoff = close - pd.Timedelta(minutes=last_entry_minutes)
        # session_close = horario do ULTIMO candle do pregao (onde zera tudo), igual ao historico
        self.session_close = session_close
        self.is_last = (df["time"] >= close).to_numpy() if session_close else None
        self.can_enter = (df["time"] <= cutoff).to_numpy()

        self.costs, self.reward_scale, self.random_days = costs, reward_scale, random_days
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-CLIP, CLIP,shape=(n_features(config.feature_set),), dtype=np.float32)
        self._next_day = 0
        self.trades: list[Trade] = []
        self.reset(seed=seed)

    # ---------------------------------------------------------------- helpers
    def _obs(self):
        i = self.i
        pf = position_features(self.pos, self.entry, self.c[i], self.atr[i], self.bars_in_pos,
                               self.config.max_hold or 120)
        return np.concatenate([self.feats[i], pf]).astype(np.float32)

    def _open(self, side, j):
        self.cur = Trade(self.day, side, self.time.iloc[j], self.o[j], None, None)
        self.pos, self.entry, self.bars_in_pos, self.entry_idx = side, self.o[j], 0, j

    def _close(self, j, price, reason):
        t = self.cur
        t.exit_time, t.exit, t.reason = self.time.iloc[j], price, reason
        t.pts = t.side * (price - t.entry) - 2 * self.costs.per_side
        self.trades.append(t)
        self.pos, self.cur, self.entry, self.bars_in_pos = 0, None, 0.0, 0

    # ---------------------------------------------------------------- gym api
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._next_day = 0  # mesma seed = mesma sequencia de pregoes
        if self.random_days:
            k = int(self.np_random.integers(len(self.days)))
        else:
            k = self._next_day % len(self.days)
            self._next_day += 1
        self.day, self.i, self.end = self.days[k]
        self.pos, self.entry, self.bars_in_pos, self.cur = 0, 0.0, 0, None
        return self._obs(), {"day": str(self.day)}

    def step(self, action):
        i, j = self.i, self.i + 1
        target = ACTION_TO_POS[int(action)]
        if self.pos != 0 and self.bars_in_pos < self.config.min_hold:
            target = self.pos  # ainda no tempo minimo: segura a posicao
        timed_out = self.pos != 0 and 0 < self.config.max_hold <= self.bars_in_pos
        if timed_out:
            target = 0  # estourou o tempo maximo: zera (nao vira a mao direto)
        if not self.can_enter[j] and target != 0 and target != self.pos:
            target = 0  # perto do fim: so pode manter ou zerar

        pnl = self.pos * (self.o[j] - self.c[i])  # gap entre o fechamento e a proxima abertura
        cost = self.costs.per_side * abs(target - self.pos)
        penalty = self.config.trade_penalty if target != 0 and target != self.pos else 0.0
        if target != self.pos:
            if self.pos != 0:
                self._close(j, self.o[j], "tempo max" if timed_out else "agente")
            if target != 0:
                self._open(target, j)
        pnl += self.pos * (self.c[j] - self.o[j])
        self.i = j
        if self.pos != 0:
            self.bars_in_pos += 1

        terminated = j >= self.end if self.is_last is None else bool(self.is_last[j])
        if terminated and self.pos != 0:
            cost += self.costs.per_side
            self._close(j, self.c[j], "fim do dia")

        reward = (pnl - cost - penalty) * self.reward_scale
        info = {"pnl_pts": pnl - cost, "gross_pts": pnl, "cost_pts": cost, "penalty_pts": penalty,
                "pos": self.pos, "timed_out": bool(timed_out)}
        return self._obs(), float(reward), terminated, False, info

    # ---------------------------------------------------------------- avaliacao
    def run_policy(self, policy) -> Result:
        """Roda policy(obs) -> acao em todos os pregoes, em ordem, e devolve os trades."""
        self.random_days, self._next_day, self.trades = False, 0, []
        for _ in range(len(self.days)):
            obs, _ = self.reset()
            done = False
            while not done:
                obs, _, done, _, _ = self.step(policy(obs))
        return Result(list(self.trades))
