"""Uma rodada de estudo de UM trader do time (o run.py roda os 5 em paralelo).

1. o aprendiz (learner.zip) continua estudando PASSOS_POR_RODADA nos pregoes de treino;
2. prova: aprendiz e campeao jogam os pregoes de validacao (que nenhum dos dois estudou);
3. se o aprendiz foi melhor, vira o novo campeao (o que vota no Megazord). Senao o campeao fica.
O aprendiz segue estudando na proxima rodada de qualquer jeito.

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
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

from winrl.data import load, split  # noqa: E402
from winrl.env import WinDayTradeEnv  # noqa: E402
from winrl.team import Trader, load_settings, probs_fn  # noqa: E402

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


train_df, val_df, _ = split(load(s.simbolo))
venv = make_vec_env(lambda: WinDayTradeEnv(train_df, s.env), n_envs=4, seed=a.trader * 1000 + a.round)
if tr.learner.exists():
    model = PPO.load(tr.learner, env=venv, device="cpu")
else:
    model = PPO("MlpPolicy", venv, seed=a.trader, n_steps=2048, batch_size=1024, learning_rate=3e-4, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, policy_kwargs=dict(net_arch=s.rede), device="cpu")

progress(phase="estudando", steps_done=0, total_timesteps=model.num_timesteps)
model.learn(total_timesteps=steps, callback=Progress(), reset_num_timesteps=False, progress_bar=False)
model.save(tr.learner)

# ---------------------------------------------------------------- prova na validacao
progress(phase="prova", steps_done=steps, total_timesteps=model.num_timesteps)
val = WinDayTradeEnv(val_df, s.env, random_days=False)


def score(path):
    f = probs_fn(PPO.load(path, device="cpu"))
    r = val.run_policy(lambda o: int(np.argmax(f(o))))
    df = r.frame()
    pts = float(df["pts"].sum()) if len(df) else 0.0
    return {"pts": round(pts, 1), "trades": len(df), "pts_dia": round(pts / len(val.days), 1),
            "acerto": round(float((df["pts"] > 0).mean()), 3) if len(df) else None}


learner = score(tr.learner)
champion = score(tr.champion) if tr.champion.exists() else None
promoted = champion is None or learner["pts"] > champion["pts"]
if promoted:
    shutil.copy(tr.learner, tr.champion)
    versions = tr.dir / "versions"
    versions.mkdir(exist_ok=True)
    shutil.copy(tr.learner, versions / f"champion_round_{a.round}.zip")

row = {"round": a.round, "time": time.strftime("%Y-%m-%d %H:%M"), "total_timesteps": model.num_timesteps,
       "learner": learner, "champion_before": champion, "promoted": promoted,
       "val_days": len(val.days), "val_period": f"{val_df['time'].min():%Y-%m-%d} a {val_df['time'].max():%Y-%m-%d}"}
tr.append_history(row)
progress(phase="pronto", steps_done=steps, total_timesteps=model.num_timesteps, learner=learner,
         champion=learner if promoted else champion, promoted=promoted)
print(json.dumps(row))
