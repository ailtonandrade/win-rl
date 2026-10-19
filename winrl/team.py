"""O time fixo de traders e o Megazord (votacao).

team/
  team.json                  historico das rodadas do time
  trader_1/ ... trader_5/
    learner.zip              o cerebro que continua estudando (nunca para)
    champion.zip             o melhor cerebro ja visto na validacao: e ele que VOTA
    history.json             uma linha por rodada (passos, nota do aprendiz, nota do campeao, promovido?)
    progress.json            andamento da rodada atual (para o painel)

Cada trader aprende sozinho. O Megazord nao aprende: ele so junta os votos dos campeoes.
"""
import configparser
import copy
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .backtest import POINT_VALUE_BRL, Result
from .env import ACTION_TO_POS, EnvConfig, WinDayTradeEnv

ROOT = Path(__file__).resolve().parents[1]
TEAM_DIR = ROOT / "team"
NAMES = ["Alfa", "Bravo", "Charlie", "Delta", "Eco", "Foxtrot", "Golf", "Hotel", "India", "Juliet"]
POS_TO_ACTION = {0: 0, 1: 1, -1: 2}


# ----------------------------------------------------------------------------- configuracao
@dataclass
class Settings:
    modo: str = "treino"
    traders: int = 5
    votos_para_entrar: int = 4
    votos_para_manter: int = 2
    simbolo: str = "WIN$N"
    fechamento: str = "18:24"  # horario do ULTIMO candle do pregao
    max_hold: int = 15
    penalidade: float = 5.0
    min_hold: int = 0
    features: str = "full"
    rede: list = field(default_factory=lambda: [128, 128])
    passos_por_rodada: int = 500_000
    rodadas: int = 0
    atualizar_dados: bool = True
    dias_teste_painel: int = 60
    conectar_mt5: bool = False
    enviar_ordens: bool = False
    velocidade_simulada: float = 2.0
    porta: int = 8765
    pasta_time: str = "team"
    abrir_navegador: bool = True
    pagina: str = "escritorio"

    @property
    def team_dir(self) -> Path:
        return ROOT / self.pasta_time

    @property
    def env(self) -> EnvConfig:
        return EnvConfig(feature_set=self.features, trade_penalty=self.penalidade,
                         min_hold=self.min_hold, max_hold=self.max_hold)


def load_settings(path: Path = ROOT / "megazord.ini") -> Settings:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cp.optionxform = str.upper
    cp.read(path, encoding="utf-8")
    flat = {k.lower(): v.strip() for sec in cp.sections() for k, v in cp[sec].items()}
    s = Settings()
    for name, default in vars(Settings()).items():
        if name not in flat:
            continue
        raw = flat[name]
        if isinstance(default, bool):
            val = raw.lower() in ("1", "true", "sim", "yes", "on")
        elif isinstance(default, int):
            val = int(raw)
        elif isinstance(default, float):
            val = float(raw)
        elif isinstance(default, list):
            val = [int(x) for x in raw.replace(",", " ").split()]
        else:
            val = raw
        setattr(s, name, val)
    if s.modo not in ("treino", "replay", "ao_vivo"):
        raise SystemExit(f"MODO invalido: {s.modo} (use treino, replay ou ao_vivo)")
    if not 1 <= s.votos_para_manter <= s.votos_para_entrar <= s.traders:
        raise SystemExit("precisa: 1 <= VOTOS_PARA_MANTER <= VOTOS_PARA_ENTRAR <= TRADERS")
    return s


# ----------------------------------------------------------------------------- traders
class Trader:
    def __init__(self, idx: int, base: Path = TEAM_DIR):
        self.idx = idx
        self.name = NAMES[idx] if idx < len(NAMES) else f"T{idx + 1}"
        self.dir = base / f"trader_{idx + 1}"
        self.learner = self.dir / "learner.zip"
        self.champion = self.dir / "champion.zip"

    @property
    def history(self) -> list:
        f = self.dir / "history.json"
        return json.loads(f.read_text()) if f.exists() else []

    def append_history(self, row: dict):
        h = self.history + [row]
        (self.dir / "history.json").write_text(json.dumps(h, indent=1))

    def progress(self) -> dict:
        f = self.dir / "progress.json"
        try:
            return json.loads(f.read_text())
        except Exception:
            return {}

    def load_champion(self):
        from stable_baselines3 import PPO
        return PPO.load(self.champion, device="cpu")


def team(s: Settings) -> list[Trader]:
    return [Trader(i, s.team_dir) for i in range(s.traders)]


def seed_team(s: Settings, sources: list[Path]) -> list[str]:
    """Cria o time a partir de modelos ja treinados (um por trader). So preenche quem esta vazio."""
    msgs = []
    for tr, src in zip(team(s), sources):
        tr.dir.mkdir(parents=True, exist_ok=True)
        if tr.learner.exists():
            continue
        shutil.copy(src, tr.learner)
        shutil.copy(src, tr.champion)
        msgs.append(f"{tr.name} <- {src.parent.name}/{src.name}")
    return msgs


# ----------------------------------------------------------------------------- votacao
def megazord_action(wants: list[int], mz_pos: int, enter: int, hold: int) -> int:
    """wants: posicao desejada de cada trader (-1, 0, +1). Devolve a acao do Megazord (0, 1, 2).
    - zerado: entra se >= enter quiserem a mesma direcao;
    - posicionado: vira a mao se >= enter quiserem o lado oposto; mantem se >= hold ainda quiserem
      o mesmo lado; senao zera."""
    n_long, n_short = wants.count(1), wants.count(-1)
    if mz_pos == 0:
        target = 1 if n_long >= enter else -1 if n_short >= enter else 0
    else:
        same, opp = (n_long, n_short) if mz_pos > 0 else (n_short, n_long)
        target = -mz_pos if opp >= enter else mz_pos if same >= hold else 0
    return POS_TO_ACTION[target]


def probs_fn(model):
    """Probabilidades das 3 acoes, em NumPy puro com os pesos da politica (MLP pequena):
    dezenas de vezes mais rapido que chamar o PyTorch candle a candle."""
    layers = []
    for m in list(model.policy.mlp_extractor.policy_net) + [model.policy.action_net]:
        if isinstance(m, torch.nn.Linear):
            layers.append(("lin", m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()))
        elif isinstance(m, torch.nn.Tanh):
            layers.append(("tanh",))
        elif isinstance(m, torch.nn.ReLU):
            layers.append(("relu",))
        else:
            raise TypeError(f"camada nao suportada: {m}")

    def f(obs):
        x = np.asarray(obs, dtype=np.float32)
        for layer in layers:
            if layer[0] == "lin":
                x = x @ layer[1] + layer[2]
            elif layer[0] == "tanh":
                x = np.tanh(x)
            else:
                x = np.maximum(x, 0)
        e = np.exp(x - x.max())
        return e / e.sum()
    return f


def clone_env(base: WinDayTradeEnv) -> WinDayTradeEnv:
    """Mesmo dado (arrays compartilhados), estado proprio."""
    e = copy.copy(base)
    e.trades, e.cur, e.pos, e.entry, e.bars_in_pos = [], None, 0, 0.0, 0
    return e


# ----------------------------------------------------------------------------- gravacao
class TeamRecorder:
    """Acumula, candle a candle, tudo que o painel do time mostra."""

    def __init__(self, names, s: Settings, env: WinDayTradeEnv, start: int):
        self.names, self.s, self.env, self.start = names, s, env, start
        self.steps, self.mfe, self.mae = [], 0.0, 0.0
        self.cum = {"reward": 0.0, "pnl": 0.0, "cost": 0.0, "penalty": 0.0}
        self.tr_cum = [0.0] * len(names)
        self.next = None

    def add(self, mz: WinDayTradeEnv, reward, info, n_closed, mz_want, tr_envs, tr_infos, tr_probs, tr_wants):
        j = mz.i
        if mz.pos == 0 or mz.bars_in_pos == 1:
            self.mfe = self.mae = 0.0
        unreal = mz.pos * (mz.c[j] - mz.entry) if mz.pos else 0.0
        if mz.pos:
            fav = mz.pos * ((mz.h[j] if mz.pos > 0 else mz.l[j]) - mz.entry)
            adv = mz.pos * ((mz.l[j] if mz.pos > 0 else mz.h[j]) - mz.entry)
            self.mfe, self.mae = max(self.mfe, fav), min(self.mae, adv)
        for k, key in (("reward", None), ("pnl", "pnl_pts"), ("cost", "cost_pts"), ("penalty", "penalty_pts")):
            self.cum[k] += reward if key is None else info[key]
        trs = []
        for n, (e, inf, p, w) in enumerate(zip(tr_envs, tr_infos, tr_probs, tr_wants)):
            self.tr_cum[n] += inf["pnl_pts"]
            trs.append({"p": [round(float(x), 3) for x in p], "want": w, "pos": e.pos,
                        "unreal": round(e.pos * (e.c[j] - e.entry), 1) if e.pos else 0.0,
                        "pnl": round(self.tr_cum[n], 1), "trades": len(e.trades)})
        self.steps.append({
            "i": j - self.start, "want": mz_want, "pos": mz.pos,
            "entry": mz.entry if mz.pos else None, "entry_i": mz.entry_idx - self.start if mz.pos else None,
            "bars_in_pos": mz.bars_in_pos, "unreal": round(unreal, 1), "mfe": round(self.mfe, 1), "mae": round(self.mae, 1),
            "reward": round(reward, 4), "pnl": round(info["pnl_pts"], 1), "cost": info["cost_pts"],
            "penalty": info["penalty_pts"], "timed_out": info["timed_out"], "closed": n_closed,
            "cum_reward": round(self.cum["reward"], 3), "cum_pnl": round(self.cum["pnl"], 1),
            "cum_cost": round(self.cum["cost"], 1), "cum_penalty": round(self.cum["penalty"], 1),
            "votes": {"long": tr_wants.count(1), "short": tr_wants.count(-1), "flat": tr_wants.count(0)},
            "tr": trs,
        })

    def episode(self, mz: WinDayTradeEnv, tr_envs, last: int) -> dict:
        env, start = self.env, self.start
        idx_of = {env.time.iloc[k]: k - start for k in range(start, last + 1)}
        bars = [[env.time.iloc[k].strftime("%H:%M"), env.o[k], env.h[k], env.l[k], env.c[k]] for k in range(start, last + 1)]
        trades = [{
            "side": t.side, "entry_i": idx_of[t.entry_time], "exit_i": idx_of[t.exit_time],
            "entry_time": t.entry_time.strftime("%H:%M"), "exit_time": t.exit_time.strftime("%H:%M"),
            "entry": t.entry, "exit": t.exit, "reason": t.reason, "pts": round(t.pts, 1),
            "brl": round(t.pts * POINT_VALUE_BRL, 2),
        } for t in mz.trades]
        pts = [t["pts"] for t in trades]
        tr_sum = [{"name": n, "pnl_pts": round(sum(t.pts for t in e.trades), 1), "trades": len(e.trades),
                   "wins": sum(t.pts > 0 for t in e.trades)} for n, e in zip(self.names, tr_envs)]
        return {
            "kind": "team", "day": str(mz.day), "bars": bars, "steps": self.steps, "trades": trades,
            "next": self.next,
            "summary": {"pnl_pts": round(sum(pts), 1), "brl": round(sum(pts) * POINT_VALUE_BRL, 2),
                        "trades": len(pts), "wins": sum(p > 0 for p in pts), "reward": round(self.cum["reward"], 3),
                        "cost_pts": round(self.cum["cost"], 1), "penalty_pts": round(self.cum["penalty"], 1),
                        "traders": tr_sum},
            "team": {"names": self.names, "enter": self.s.votos_para_entrar, "hold": self.s.votos_para_manter},
            "config": mz.config.to_dict(), "point_value": POINT_VALUE_BRL, "cost_per_side": mz.costs.per_side,
        }


def play_day(base: WinDayTradeEnv, fns, names, s: Settings, day_index: int, record=True):
    """O time inteiro joga um pregao do historico. Cada trader opera a propria conta-sombra;
    o Megazord opera a conta dele com a votacao. Devolve (episodio ou None, envs)."""
    envs = [clone_env(base) for _ in range(len(fns) + 1)]
    for e in envs:
        e.random_days, e._next_day = False, day_index
        e.reset()
    mz, trs = envs[-1], envs[:-1]
    rec = TeamRecorder(names, s, base, mz.i) if record else None
    done = False
    while not done:
        probs = [f(e._obs()) for f, e in zip(fns, trs)]
        acts = [int(np.argmax(p)) for p in probs]
        wants = [ACTION_TO_POS[a] for a in acts]
        mz_act = megazord_action(wants, mz.pos, s.votos_para_entrar, s.votos_para_manter)
        infos = [e.step(a)[4] for e, a in zip(trs, acts)]
        n_before = len(mz.trades)
        _, reward, done, _, info = mz.step(mz_act)
        if rec:
            rec.add(mz, reward, info, len(mz.trades) - n_before, ACTION_TO_POS[mz_act], trs, infos, probs, wants)
    return (rec.episode(mz, trs, mz.i) if rec else None), envs


def play_days(base: WinDayTradeEnv, fns, names, s: Settings, days=None):
    """Joga varios pregoes sem gravar detalhes. Devolve Result do Megazord e de cada trader."""
    days = range(len(base.days)) if days is None else days
    mz_trades, tr_trades = [], [[] for _ in fns]
    for k in days:
        _, envs = play_day(base, fns, names, s, k, record=False)
        mz_trades += envs[-1].trades
        for n, e in enumerate(envs[:-1]):
            tr_trades[n] += e.trades
    return Result(mz_trades), [Result(t) for t in tr_trades]
