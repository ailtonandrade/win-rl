"""Grava episodios (um pregao jogado pelo agente) para o painel ao vivo em dashboard/.

dashboard/data/index.json   -> {mode, title, config, episodes: [...], metrics: [...]}
dashboard/data/ep_<n>.json  -> candles + cada passo do agente + trades do dia

O painel (dashboard/index.html) le esses arquivos e anima candle a candle.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .backtest import POINT_VALUE_BRL
from .env import ACTION_TO_POS, WinDayTradeEnv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dashboard" / "data"


def _write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=_default), encoding="utf-8")
    for _ in range(20):  # no Windows o replace falha se o navegador estiver lendo naquele instante
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def action_probs(model, obs):
    with torch.no_grad():
        t, _ = model.policy.obs_to_tensor(obs)
        return model.policy.get_distribution(t).distribution.probs[0].cpu().numpy()


def record_episode(env: WinDayTradeEnv, model, day_index: int, deterministic=True) -> dict:
    """Joga um pregao inteiro e devolve tudo que aconteceu, passo a passo."""
    env.random_days, env._next_day, env.trades = False, day_index, []
    obs, _ = env.reset()
    start = env.i
    steps, cum_reward, cum_pnl, cum_cost, cum_pen = [], 0.0, 0.0, 0.0, 0.0
    mfe = mae = 0.0
    done = False
    while not done:
        probs = action_probs(model, obs)
        action = int(np.argmax(probs)) if deterministic else int(np.random.choice(3, p=probs))
        n_before = len(env.trades)
        obs, reward, done, _, info = env.step(action)
        j = env.i
        closed = env.trades[n_before:]
        if env.pos == 0 or env.bars_in_pos == 1:
            mfe = mae = 0.0  # zerado, ou trade aberto neste candle
        unreal = env.pos * (env.c[j] - env.entry) if env.pos else 0.0
        if env.pos:
            fav = env.pos * ((env.h[j] if env.pos > 0 else env.l[j]) - env.entry)
            adv = env.pos * ((env.l[j] if env.pos > 0 else env.h[j]) - env.entry)
            mfe, mae = max(mfe, fav), min(mae, adv)
        cum_reward += reward
        cum_pnl += info["pnl_pts"]
        cum_cost += info["cost_pts"]
        cum_pen += info["penalty_pts"]
        steps.append({
            "i": j - start, "want": ACTION_TO_POS[action], "probs": [round(float(p), 3) for p in probs],
            "pos": env.pos, "entry": env.entry if env.pos else None,
            "entry_i": env.entry_idx - start if env.pos else None,
            "bars_in_pos": env.bars_in_pos, "unreal": round(unreal, 1), "mfe": round(mfe, 1), "mae": round(mae, 1),
            "reward": round(reward, 4), "pnl": round(info["pnl_pts"], 1), "cost": info["cost_pts"],
            "penalty": info["penalty_pts"], "timed_out": info["timed_out"],
            "cum_reward": round(cum_reward, 3), "cum_pnl": round(cum_pnl, 1),
            "cum_cost": round(cum_cost, 1), "cum_penalty": round(cum_pen, 1),
            "closed": len(closed),
        })
    sl = slice(start, env.end + 1)
    bars = [[env.time.iloc[k].strftime("%H:%M"), env.o[k], env.h[k], env.l[k], env.c[k]] for k in range(sl.start, sl.stop)]
    idx_of = {env.time.iloc[k]: k - start for k in range(sl.start, sl.stop)}
    trades = [{
        "side": t.side, "entry_i": idx_of[t.entry_time], "exit_i": idx_of[t.exit_time],
        "entry_time": t.entry_time.strftime("%H:%M"), "exit_time": t.exit_time.strftime("%H:%M"),
        "entry": t.entry, "exit": t.exit, "reason": t.reason, "pts": round(t.pts, 1),
        "brl": round(t.pts * POINT_VALUE_BRL, 2),
    } for t in env.trades]
    pts = [t["pts"] for t in trades]
    return {
        "day": str(env.day), "bars": bars, "steps": steps, "trades": trades,
        "summary": {"pnl_pts": round(sum(pts), 1), "brl": round(sum(pts) * POINT_VALUE_BRL, 2),
                    "trades": len(pts), "wins": sum(p > 0 for p in pts), "reward": round(cum_reward, 3),
                    "cost_pts": round(cum_cost, 1), "penalty_pts": round(cum_pen, 1)},
        "config": env.config.to_dict(), "point_value": POINT_VALUE_BRL,
        "cost_per_side": env.costs.per_side,
    }


class Board:
    """Mantem o index.json do painel."""

    def __init__(self, mode: str, title: str, config: dict, keep=60, data_dir: Path = DATA_DIR):
        self.dir = data_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        for f in self.dir.glob("*.json"):
            f.unlink()
        self.index = {"mode": mode, "title": title, "config": config, "episodes": [], "metrics": [],
                      "updated": time.time()}
        self.keep, self.n = keep, 0
        self.save()

    def add_episode(self, ep: dict, **extra):
        self.n += 1
        name = f"ep_{self.n}.json"
        _write_json(self.dir / name, ep)
        self.index["episodes"].append({"id": self.n, "file": name, "day": ep["day"], **ep["summary"], **extra})
        if len(self.index["episodes"]) > self.keep:  # em treino longo, mantem so os ultimos
            old = self.index["episodes"].pop(0)
            (self.dir / old["file"]).unlink(missing_ok=True)
        self.save()

    def add_metrics(self, row: dict):
        self.index["metrics"].append(row)
        self.save()

    def set_team(self, info: dict):
        """Status do time (progresso de cada trader, historico das rodadas) para o painel do Megazord."""
        self.index["team"] = info
        self.save()

    def upsert_live(self, ep: dict):
        """Modo ao vivo: um unico episodio (o pregao de hoje) reescrito a cada candle."""
        _write_json(self.dir / "ep_live.json", ep)
        self.index["episodes"] = [{"id": 1, "file": "ep_live.json", "day": ep["day"], **ep["summary"],
                                   "bars": len(ep["bars"])}]
        self.index["live"] = ep.get("live")
        self.save()

    def live_status(self, source: str, msg: str):
        self.index["live"] = {**(self.index.get("live") or {}), "source": source, "status": msg,
                              "updated": time.strftime("%H:%M:%S")}
        self.save()

    def save(self):
        self.index["updated"] = time.time()
        _write_json(self.dir / "index.json", self.index)


class WatchCallback(BaseCallback):
    """Durante o TREINO: a cada N passos joga um pregao da validacao com a politica atual
    e publica no painel, junto com as metricas de aprendizado."""

    def __init__(self, board: Board, val_env: WinDayTradeEnv, every=20_000):
        super().__init__()
        self.board, self.val_env, self.every = board, val_env, every
        self.next_at, self.k = 0, 0

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if self.num_timesteps < self.next_at:
            return
        self.next_at = self.num_timesteps + self.every
        ep = record_episode(self.val_env, self.model, self.k % len(self.val_env.days))
        self.k += 7  # pula dias para variar o pregao mostrado
        logs = dict(self.model.logger.name_to_value)
        buf = [e["r"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
        self.board.add_episode(ep, timesteps=self.num_timesteps)
        self.board.add_metrics({
            "timesteps": self.num_timesteps,
            "demo_pnl": ep["summary"]["pnl_pts"], "demo_trades": ep["summary"]["trades"],
            "train_reward_pts": round(float(np.mean(buf)) * 100, 1) if buf else None,
            "entropy": round(-logs["train/entropy_loss"], 4) if "train/entropy_loss" in logs else None,
            "value_loss": round(logs["train/value_loss"], 4) if "train/value_loss" in logs else None,
            "approx_kl": round(logs["train/approx_kl"], 5) if "train/approx_kl" in logs else None,
            "explained_var": round(logs["train/explained_variance"], 3) if "train/explained_variance" in logs else None,
        })
