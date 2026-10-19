"""Treina um PPO. Modo TREINO: dados historicos, ordens simuladas, nada vai para o MT5.

  python scripts/train.py --data WIN$N --steps 2000000 --seed 0
  python scripts/train.py --data momentum --steps 500000   (teste de sanidade)

Salva em models/<data>_s<seed>/: checkpoints versionados (ckpt_*.zip), best_model.zip
(melhor na VALIDACAO, nunca no teste) e logs do tensorboard.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from winrl.data import load, split  # noqa: E402
from winrl.env import EnvConfig, WinDayTradeEnv  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="WIN$N")
p.add_argument("--steps", type=int, default=2_000_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--envs", type=int, default=8)
p.add_argument("--features", default="full", choices=["full", "small"])
p.add_argument("--penalty", type=float, default=0.0, help="pts extras por entrada, so no reward")
p.add_argument("--min-hold", type=int, default=0, help="candles minimos em posicao")
p.add_argument("--max-hold", type=int, default=0, help="candles maximos em posicao (0 = sem limite)")
p.add_argument("--net", type=int, nargs="+", default=[128, 128])
p.add_argument("--tag", default="", help="nome da configuracao, vira prefixo da pasta do modelo")
p.add_argument("--threads", type=int, default=0, help="threads do torch (0 = padrao)")
p.add_argument("--resume", help="checkpoint .zip para continuar o treino de onde parou")
p.add_argument("--watch", type=int, default=0, metavar="N",
               help="publica no painel (dashboard/) um pregao jogado pela politica atual a cada N passos")
a = p.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)

cfg = EnvConfig(feature_set=a.features, trade_penalty=a.penalty, min_hold=a.min_hold, max_hold=a.max_hold)
train_df, val_df, _ = split(load(a.data))
out = ROOT / "models" / f"{a.tag + '_' if a.tag else ''}{a.data.replace('$', '_')}_s{a.seed}"
out.mkdir(parents=True, exist_ok=True)
# a avaliacao e o tempo real leem isto para montar EXATAMENTE o mesmo ambiente
(out / "config.json").write_text(json.dumps({
    "data": a.data, "env": cfg.to_dict(), "net": a.net, "steps": a.steps, "seed": a.seed,
    # periodo visto no treino/validacao: a avaliacao recusa "teste" que se sobreponha a isto
    "train_start": str(train_df["time"].min().date()), "val_end": str(val_df["time"].max().date()),
}, indent=2))
print(f"treino: {train_df['time'].dt.date.nunique()} pregoes, validacao: {val_df['time'].dt.date.nunique()} -> {out}")

venv = make_vec_env(lambda: WinDayTradeEnv(train_df, cfg), n_envs=a.envs, seed=a.seed)
val_env = Monitor(WinDayTradeEnv(val_df, cfg, random_days=False))
n_val = len(val_env.unwrapped.days)

if a.resume:
    # retoma pesos + estado do otimizador + contador de passos
    model = PPO.load(a.resume, env=venv)
else:
    model = PPO("MlpPolicy", venv, seed=a.seed, n_steps=2048, batch_size=1024, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
                policy_kwargs=dict(net_arch=a.net), tensorboard_log=str(out / "tb"), verbose=0)

callbacks = [
    CheckpointCallback(save_freq=max(250_000 // a.envs, 1), save_path=str(out), name_prefix="ckpt"),
    EvalCallback(val_env, n_eval_episodes=n_val, eval_freq=max(100_000 // a.envs, 1), deterministic=True,
                 best_model_save_path=str(out), log_path=str(out), verbose=1),
]
if a.watch:
    from winrl.live import Board, WatchCallback

    board = Board("treino", f"Treinando {out.name}", {**cfg.to_dict(), "net": a.net, "steps": a.steps})
    callbacks.append(WatchCallback(board, WinDayTradeEnv(val_df, cfg, random_days=False), every=a.watch))
model.learn(total_timesteps=a.steps, callback=callbacks, reset_num_timesteps=not a.resume, progress_bar=False)
model.save(out / "last_model")
print("ok", out)
