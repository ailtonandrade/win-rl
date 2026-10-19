"""Uma rodada de estudo de UM trader do time (o run.py roda os 5 em paralelo).

1. o aprendiz (learner.zip) continua estudando PASSOS_POR_RODADA nos pregoes de treino,
   com uma taxa de aprendizado propria de cada trader (diversidade);
   se falhou RESET_APRENDIZ_APOS rodadas seguidas, antes recomeca do campeao com passos menores;
2. prova: aprendiz e campeao jogam os pregoes de validacao (que nenhum dos dois estudou);
3. promocao por merito: compara o resultado DIA A DIA (teste t pareado). So promove se o
   aprendiz for melhor na media e com t >= PROMOCAO_T_MIN. Senao o campeao fica.

  python scripts/team_round.py --trader 0 --round 3
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

from winrl.env import WinDayTradeEnv  # noqa: E402
from stable_baselines3.common.utils import get_schedule_fn  # noqa: E402

from winrl.team import Trader, load_settings, probs_fn, splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--trader", type=int, required=True)
p.add_argument("--round", type=int, required=True)
p.add_argument("--steps", type=int, help="sobrescreve PASSOS_POR_RODADA")
p.add_argument("--ini", default=str(ROOT / "megazord.ini"))
a = p.parse_args()

torch.set_num_threads(1)
s = load_settings(Path(a.ini))
tr = Trader(a.trader, s.team_dir)
tr.dir.mkdir(parents=True, exist_ok=True)
steps = a.steps or s.passos_por_rodada


def progress(**kw):
    (tr.dir / "progress.json").write_text(json.dumps({"round": a.round, "name": tr.name, "steps_total": steps,
                                                      "time": time.time(), **kw}))


class Progress(BaseCallback):
    def __init__(self):
        super().__init__()
        self.start = None

    def _on_step(self):
        if self.start is None:
            self.start = self.num_timesteps
        if self.n_calls % 500 == 0:
            progress(phase="estudando", steps_done=self.num_timesteps - self.start, total_timesteps=self.num_timesteps)
        return True


LR_BASE = 3e-4
LR_MULT = [1.0, .7, .5, .35, .25]  # cada trader estuda num ritmo diferente

train_df, val_df, _ = splits(s)
hist = tr.history
fails = 0
for r in reversed(hist):
    if r.get("promoted"):
        break
    fails += 1
reset = bool(s.reset_aprendiz_apos) and fails >= s.reset_aprendiz_apos and tr.champion.exists()
if reset:
    # travado: o aprendiz recomeca do melhor cerebro ja testado e refina com passos menores
    shutil.copy(tr.champion, tr.learner)
lr = LR_BASE * LR_MULT[a.trader % len(LR_MULT)] * (.5 if reset else 1.0)

venv = make_vec_env(lambda: WinDayTradeEnv(train_df, s.env), n_envs=4, seed=a.trader * 1000 + a.round)
if tr.learner.exists():
    model = PPO.load(tr.learner, env=venv, device="cpu")
    model.learning_rate = lr
    model.lr_schedule = get_schedule_fn(lr)
else:
    model = PPO("MlpPolicy", venv, seed=a.trader, n_steps=2048, batch_size=1024, learning_rate=lr, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, policy_kwargs=dict(net_arch=s.rede), device="cpu")
if reset:
    model.set_random_seed(a.trader * 7919 + a.round)

progress(phase="estudando", steps_done=0, total_timesteps=model.num_timesteps)
model.learn(total_timesteps=steps, callback=Progress(), reset_num_timesteps=False, progress_bar=False)
model.save(tr.learner)

# ---------------------------------------------------------------- prova na validacao
progress(phase="prova", steps_done=steps, total_timesteps=model.num_timesteps)
val = WinDayTradeEnv(val_df, s.env, random_days=False)
all_days = [d for d, _, _ in val.days]


def score(path):
    """Resultado na validacao: total e o PnL de cada pregao (para comparar dia a dia)."""
    f = probs_fn(PPO.load(path, device="cpu"))
    r = val.run_policy(lambda o: int(np.argmax(f(o))))
    df = r.frame()
    daily = df.groupby("day")["pts"].sum() if len(df) else pd.Series(dtype=float)
    daily = np.array([float(daily.get(d, 0.0)) for d in all_days])
    pts = float(daily.sum())
    return {"pts": round(pts, 1), "trades": len(df), "pts_dia": round(pts / len(all_days), 1),
            "acerto": round(float((df["pts"] > 0).mean()), 3) if len(df) else None}, daily


learner, l_daily = score(tr.learner)
if tr.champion.exists():
    champion, c_daily = score(tr.champion)
    diff = l_daily - c_daily
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    t_stat = float(diff.mean() / (sd / np.sqrt(len(diff)))) if sd > 0 else 0.0
    promoted = bool(diff.mean() > 0 and t_stat >= s.promocao_t_min)
else:
    champion, diff, t_stat, promoted = None, np.zeros(len(all_days)), None, True
if promoted:
    shutil.copy(tr.learner, tr.champion)
    versions = tr.dir / "versions"
    versions.mkdir(exist_ok=True)
    shutil.copy(tr.learner, versions / f"champion_round_{a.round}.zip")

row = {"round": a.round, "time": time.strftime("%Y-%m-%d %H:%M"), "total_timesteps": model.num_timesteps,
       "learner": learner, "champion_before": champion, "promoted": promoted,
       "t_stat": None if t_stat is None else round(t_stat, 2),
       "diff_pts_dia": round(float(diff.mean()), 1), "dias_melhor": int((diff > 0).sum()), "dias_pior": int((diff < 0).sum()),
       "lr": lr, "reset": reset,
       "val_days": len(all_days), "val_period": f"{val_df['time'].min():%Y-%m-%d} a {val_df['time'].max():%Y-%m-%d}"}
tr.append_history(row)
progress(phase="pronto", steps_done=steps, total_timesteps=model.num_timesteps, learner=learner,
         champion=learner if promoted else champion, promoted=promoted, t_stat=row["t_stat"], reset=reset)
print(json.dumps(row))
