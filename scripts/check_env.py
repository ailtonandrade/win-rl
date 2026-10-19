"""Checagens do ambiente: a soma dos rewards tem que bater com o PnL dos trades,
e um agente que nunca opera tem que dar exatamente zero."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from gymnasium.utils.env_checker import check_env  # noqa: E402

from winrl.data import load  # noqa: E402
from winrl.env import WinDayTradeEnv  # noqa: E402

for source in ("WIN$N", "momentum"):
    env = WinDayTradeEnv(load(source), random_days=False)
    check_env(env, skip_render_check=True)
    rng = np.random.default_rng(0)
    total_reward, env.trades = 0.0, []
    for _ in range(10):
        env.reset()
        done = False
        while not done:
            _, r, done, _, info = env.step(int(rng.integers(3)))
            total_reward += info["pnl_pts"]
    trades_pts = sum(t.pts for t in env.trades)
    flat = env.run_policy(lambda o: 0).summary()
    print(f"{source}: rewards={total_reward:.1f} trades={trades_pts:.1f} "
          f"bate={abs(total_reward - trades_pts) < 1e-6}  nunca_opera={flat['trades']} trades")
