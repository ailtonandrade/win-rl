"""Features do agente. O MESMO codigo serve treino, backtest e tempo real.

Tudo e calculado so com informacao disponivel no fechamento do candle i e em
escala relativa (dividido pelo ATR / medias moveis), entao nao precisa de
estatisticas de normalizacao salvas a parte.
"""
import numpy as np
import pandas as pd

RET_LAGS = 30   # retornos dos ultimos N candles
ATR_N = 20
VOL_N = 60
CLIP = 5.0

MARKET_FEATURES = (
    [f"ret_{k}" for k in range(RET_LAGS)]
    + ["body", "range", "upper_wick", "lower_wick", "vol_rel", "dist_day_open", "dist_vwap",
       "dist_day_high", "dist_day_low", "time_of_day", "bars_left"]
)
POSITION_FEATURES = ["pos", "unrealized", "bars_in_pos"]

# "small": menos coisa para decorar. 5 retornos em vez de 30, e so o contexto do dia.
FEATURE_SETS = {
    "full": MARKET_FEATURES,
    "small": [f"ret_{k}" for k in range(5)]
    + ["range", "vol_rel", "dist_day_open", "dist_vwap", "time_of_day", "bars_left"],
}


def n_features(feature_set="full"):
    return len(FEATURE_SETS[feature_set]) + len(POSITION_FEATURES)


def market_features(df: pd.DataFrame, feature_set="full", session_close: str | None = None) -> pd.DataFrame:
    """df canonico (time, open, high, low, close, volume), varios pregoes em sequencia.
    session_close "HH:MM": horario do ULTIMO candle do pregao, conhecido de antemao (tempo real). Sem ele,
    usa o ultimo candle de cada dia (so da para saber no historico)."""
    df = df.reset_index(drop=True)
    day = df["time"].dt.date
    rng = df["high"] - df["low"]
    atr = rng.rolling(ATR_N, min_periods=1).mean().clip(lower=5.0)

    f = pd.DataFrame(index=df.index)
    ret = df["close"].diff().fillna(0.0) / atr
    for k in range(RET_LAGS):
        f[f"ret_{k}"] = ret.shift(k).fillna(0.0)
    f["body"] = (df["close"] - df["open"]) / atr
    f["range"] = rng / atr
    f["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / atr
    f["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / atr
    vol_mean = df["volume"].rolling(VOL_N, min_periods=1).mean().replace(0, np.nan)
    f["vol_rel"] = np.log((df["volume"] / vol_mean).fillna(1.0).clip(lower=1e-3))

    g = df.groupby(day)
    day_open = g["open"].transform("first")
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum().replace(0, np.nan)
    f["dist_day_open"] = (df["close"] - day_open) / atr
    f["dist_vwap"] = ((df["close"] - vwap) / atr).fillna(0.0)
    f["dist_day_high"] = (g["high"].cummax() - df["close"]) / atr
    f["dist_day_low"] = (df["close"] - g["low"].cummin()) / atr

    minutes = df["time"].dt.hour * 60 + df["time"].dt.minute
    f["time_of_day"] = (minutes - 9 * 60) / (9.5 * 60)  # ~0 na abertura, ~1 no fechamento
    # minutos ate o fim do pregao. O horario de fechamento muda (17:54 em 2022-23, 18:24 depois),
    # entao no historico usa o ultimo candle de cada dia; em tempo real deve vir da grade da B3.
    if session_close:
        hh, mm = map(int, session_close.split(":"))
        end_min = hh * 60 + mm
    else:
        end_min = g["time"].transform("max")
        end_min = end_min.dt.hour * 60 + end_min.dt.minute
    f["bars_left"] = (end_min - minutes).clip(lower=0) / 560

    return f[FEATURE_SETS[feature_set]].clip(-CLIP, CLIP).astype(np.float32)


def position_features(pos: int, entry: float, close: float, atr: float, bars_in_pos: int,
                      hold_scale: int = 120) -> np.ndarray:
    """hold_scale = tempo maximo em posicao quando existe, entao 1.0 = "vai ser zerado agora"."""
    unreal = pos * (close - entry) / atr if pos != 0 else 0.0
    return np.array([pos, np.clip(unreal, -CLIP, CLIP), min(bars_in_pos, hold_scale) / hold_scale], dtype=np.float32)
