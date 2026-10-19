"""Modo AO VIVO do Emillion: o time olha cada candle de M1 assim que ele fecha e vota.

Paper trading: as "ordens" sao simuladas com as mesmas regras e custos do treino
(decide no fechamento, executa na abertura do candle seguinte). Nada e enviado ao MT5.

Fontes de candles:
- MT5Feed: le o WIN$N do terminal aberto (conta DEMO), so candles ja fechados;
- SimFeed: reproduz um pregao real do historico no relogio (para testar com o mercado fechado).
"""
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .env import ACTION_TO_POS, WinDayTradeEnv
from .team import Settings, TeamRecorder, clone_env, megazord_action

CONTEXT_DAYS = 2  # pregoes anteriores carregados para as medias moveis (ATR, volume)


# ----------------------------------------------------------------------------- fontes
class SimFeed:
    """Finge que `day` esta acontecendo agora, revelando `speed` candles por segundo."""

    def __init__(self, df: pd.DataFrame, speed: float, day=None):
        dates = sorted(df["time"].dt.date.unique())
        self.day = day or dates[-1]
        k = dates.index(self.day)
        keep = set(dates[max(0, k - CONTEXT_DAYS): k + 1])
        self.df = df[df["time"].dt.date.isin(keep)].reset_index(drop=True)
        self.first_today = int((self.df["time"].dt.date == self.day).idxmax())
        self.speed, self.t0 = speed, time.time()
        self.label = f"SIMULADO: pregao de {self.day} a {speed:g} candles/s"

    def closed_bars(self):
        n = self.first_today + 1 + int((time.time() - self.t0) * self.speed)
        return self.df.iloc[:min(n, len(self.df))]

    def finished(self):
        return self.first_today + int((time.time() - self.t0) * self.speed) >= len(self.df)


class MT5Feed:
    def __init__(self, symbol: str, allow_real=False):
        import MetaTrader5 as mt5
        self.mt5, self.symbol = mt5, symbol
        if not mt5.initialize():
            raise SystemExit(f"Nao conectou no MT5: {mt5.last_error()} (o terminal esta aberto e logado?)")
        acc = mt5.account_info()
        if acc is None or (acc.trade_mode != 0 and not allow_real):
            raise SystemExit("O MT5 nao esta numa conta DEMO. Por padrao o Emillion so roda em conta demo "
                             "(PERMITIR_CONTA_REAL no megazord.ini muda isso).")
        if not mt5.symbol_select(symbol, True):
            raise SystemExit(f"Simbolo {symbol} nao encontrado no MT5")
        self.label = f"MT5 conectado: {acc.server} ({'CONTA REAL' if acc.trade_mode != 0 else 'conta demo'} {acc.login})"
        self.ctx_start, self.ctx_day = None, None

    def closed_bars(self):
        mt5 = self.mt5
        r = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 2500)
        tick = mt5.symbol_info_tick(self.symbol)
        if r is None or len(r) == 0 or tick is None:
            return None
        df = pd.DataFrame(r)
        # o ultimo candle ainda esta se formando enquanto o relogio do servidor nao passar do minuto dele
        df = df[df["time"] + 60 <= tick.time]
        self.server_epoch = int(tick.time)
        self.last_closed_epoch = int(df["time"].iloc[-1]) if len(df) else 0
        df = pd.DataFrame({
            "time": pd.to_datetime(df["time"], unit="s").dt.tz_localize("America/Sao_Paulo"),
            "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
            "volume": df["real_volume"], "trades": df["tick_volume"], "spread": df["spread"],
        }).reset_index(drop=True)
        dates = sorted(df["time"].dt.date.unique())
        today = dates[-1]
        if today != self.ctx_day:  # inicio fixo durante o dia, para os indices nao andarem
            self.ctx_day = today
            self.ctx_start = dates[max(0, len(dates) - 1 - CONTEXT_DAYS)]
        return df[df["time"].dt.date >= self.ctx_start].reset_index(drop=True)

    def forming(self) -> dict | None:
        """Preco agora e o candle de 1 minuto que ainda esta se formando (so para as telas; nao decide nada)."""
        mt5 = self.mt5
        tick = mt5.symbol_info_tick(self.symbol)
        r = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 1)
        if tick is None or r is None or len(r) == 0:
            return None
        bar = r[-1]
        price = float(tick.last or tick.bid or bar["close"])
        to_hm = lambda ep: datetime.fromtimestamp(int(ep), tz=timezone.utc).strftime("%H:%M")  # noqa: E731
        return {"t": to_hm(bar["time"]), "o": float(bar["open"]), "h": max(float(bar["high"]), price),
                "l": min(float(bar["low"]), price), "c": price, "bid": float(tick.bid), "ask": float(tick.ask),
                "server": datetime.fromtimestamp(int(tick.time), tz=timezone.utc).strftime("%H:%M:%S"),
                "forming": int(bar["time"]) + 60 > int(tick.time)}

    def is_fresh(self, max_late=90):
        """O ultimo candle fechou agora ha pouco? (evita mandar ordem por candles antigos, ex.: na recuperacao da manha)"""
        return getattr(self, "server_epoch", 0) - (getattr(self, "last_closed_epoch", 0) + 60) <= max_late

    def server_time(self):
        tick = self.mt5.symbol_info_tick(self.symbol)
        return datetime.utcfromtimestamp(tick.time) if tick else None

    def finished(self):
        return False


# ----------------------------------------------------------------------------- estado de cada conta
class Book:
    """Posicao de uma conta (trader-sombra ou Emillion) entre um candle e outro."""

    def __init__(self):
        self.pos, self.entry, self.entry_time, self.bars_in_pos, self.cur = 0, 0.0, None, 0, None
        self.trades, self.pending = [], None

    def into(self, base: WinDayTradeEnv, i: int, today, end: int) -> WinDayTradeEnv:
        e = clone_env(base)
        e.i, e.day, e.end, e.trades = i, today, end, self.trades
        e.pos, e.entry, e.bars_in_pos, e.cur = self.pos, self.entry, self.bars_in_pos, self.cur
        if self.pos:
            e.entry_idx = int(np.flatnonzero(base.time.to_numpy() == self.entry_time)[0])
        return e

    def back(self, e: WinDayTradeEnv):
        self.pos, self.entry, self.bars_in_pos, self.cur = e.pos, e.entry, e.bars_in_pos, e.cur
        self.entry_time = e.cur.entry_time if e.cur is not None else None


class LiveTeam:
    def __init__(self, fns, names, s: Settings):
        self.fns, self.names, self.s = fns, names, s
        self.day, self.last_j, self.rec, self.closed_day = None, None, None, False

    def _new_day(self, base, today, start):
        self.day, self.last_j, self.closed_day = today, None, False
        self.books = [Book() for _ in range(len(self.fns) + 1)]
        self.rec = TeamRecorder(self.names, self.s, base, start)
        self.mz_env = self.tr_envs = None

    def update(self, df: pd.DataFrame):
        """Processa os candles fechados novos. Devolve o episodio do dia (ou None se nada mudou)."""
        if df is None or df.empty:
            return None
        today = df["time"].dt.date.iloc[-1]
        base = WinDayTradeEnv(df, self.s.env, random_days=False, session_close=self.s.fechamento)
        idx_today = np.flatnonzero((df["time"].dt.date == today).to_numpy())
        start, last = int(idx_today[0]), int(idx_today[-1])
        if today != self.day:
            self._new_day(base, today, start)
        if self.last_j == last or self.closed_day:
            return None
        self.rec.env = base
        first_new = start if self.last_j is None else self.last_j + 1
        for j in range(first_new, last + 1):
            self._bar(base, today, start, last, j)
            if self.closed_day:
                break
        self.last_j = last
        envs = [b.into(base, last, today, last) for b in self.books]
        return self.rec.episode(envs[-1], envs[:-1], last)

    def real_target(self, last_time) -> int:
        """Posicao que a conta real deve ter na abertura do proximo candle: a decisao do Emillion
        com as mesmas regras do treino (tempo minimo/maximo, sem entrar perto do fim) e zerando
        ZERAR_ANTES_MIN antes do ultimo candle (a corretora nao deixa passar do pregao)."""
        b = self.books[-1] if self.day is not None else None
        if b is None or b.pending is None or self.closed_day:
            return 0
        target, cfg = ACTION_TO_POS[b.pending], self.s.env
        if b.pos != 0 and b.bars_in_pos < cfg.min_hold:
            target = b.pos
        if b.pos != 0 and 0 < cfg.max_hold <= b.bars_in_pos:
            target = 0
        hh, mm = map(int, self.s.fechamento.split(":"))
        close = last_time.normalize() + pd.Timedelta(hours=hh, minutes=mm)
        nxt = last_time + pd.Timedelta(minutes=1)
        if nxt > close - pd.Timedelta(minutes=30) and target != 0 and target != b.pos:
            target = 0
        if nxt >= close - pd.Timedelta(minutes=self.s.zerar_antes_min):
            target = 0
        return target

    def _bar(self, base, today, start, last, j):
        books = self.books
        # 1) executa o que foi decidido no candle anterior (na abertura deste) e marca a mercado
        if all(b.pending is not None for b in books) and j > start:
            envs = [b.into(base, j - 1, today, last) for b in books]
            infos, rewards = [], []
            n_before = len(books[-1].trades)
            for e, b in zip(envs, books):
                _, r, done, _, info = e.step(b.pending)
                infos.append(info)
                rewards.append(r)
                b.back(e)
            mz, trs = envs[-1], envs[:-1]
            self.rec.add(mz, rewards[-1], infos[-1], len(mz.trades) - n_before, ACTION_TO_POS[books[-1].pending],
                         trs, infos[:-1], self._last_probs, self._last_wants)
            if done:
                self.closed_day = True
                self.rec.next = None
                return
        # 2) decide no fechamento deste candle (vale para a abertura do proximo)
        envs = [b.into(base, j, today, last) for b in books]
        probs = [f(e._obs()) for f, e in zip(self.fns, envs[:-1])]
        acts = [int(np.argmax(p)) for p in probs]
        wants = [ACTION_TO_POS[a] for a in acts]
        mz_act = megazord_action(wants, books[-1].pos, self.s.votos_para_entrar, self.s.votos_para_manter)
        for b, a in zip(books, acts + [mz_act]):
            b.pending = a
        self._last_probs, self._last_wants = probs, wants
        self.rec.next = {"i": j - start, "probs": [[round(float(x), 3) for x in p] for p in probs], "wants": wants,
                         "mz_want": ACTION_TO_POS[mz_act],
                         "votes": {"long": wants.count(1), "short": wants.count(-1), "flat": wants.count(0)}}


def run_live(feed, fns, names, s: Settings, board, poll=1.0, broker=None):
    """Loop principal do modo ao vivo. Publica o dia no painel a cada candle novo.
    Com `broker` (ENVIAR_ORDENS = true, so conta demo) a posicao do Emillion vira ordem no MT5."""
    team = LiveTeam(fns, names, s)
    print(feed.label)
    if broker:
        print(f"ORDENS LIGADAS: {broker.label}. Ctrl+C zera a posicao e para.")
        problem = broker.check_trading_enabled()
        if problem:
            print(f"!! ATENCAO: {problem}. As ordens vao ser recusadas ate isso ser ligado.")
    else:
        print("Paper trading: NENHUMA ordem e enviada. Ctrl+C para parar.")
    last_status = 0
    while True:
        df = feed.closed_bars()
        ep = team.update(df)
        if broker:
            # so age em candle recem-fechado; na recuperacao da manha (candles antigos) nao manda nada
            if ep is not None and getattr(feed, "is_fresh", lambda: True)():
                votes = (ep.get("next") or {}).get("votes", {})
                target = team.real_target(df["time"].iloc[-1])
                paper = team.books[-1].pos
                if broker.position() == 0 and paper != 0 and target == paper:
                    target = 0  # o trade do Emillion comecou antes (ex.: na recuperacao da manha): nao entra no meio
                broker.sync(target, f"C{votes.get('long', 0)} V{votes.get('short', 0)}")
            elif team.closed_day or broker._guards():
                broker.sync(0, "fim do dia" if team.closed_day else "trava")
        if ep is not None:
            ep["live"] = {"source": feed.label, "updated": datetime.now().strftime("%H:%M:%S"),
                          "closed": team.closed_day, "broker": broker.status() if broker else None}
            board.upsert_live(ep)
            pos = ep["steps"][-1]["pos"] if ep["steps"] else 0
            votes = (ep.get("next") or {}).get("votes", {})
            print(f"{ep['bars'][-1][0]}  preco {ep['bars'][-1][4]:,.0f}  votos compra {votes.get('long', '-')} "
                  f"venda {votes.get('short', '-')}  megazord {('ZERADO', 'COMPRADO', 'VENDIDO')[pos]}"
                  f"  dia {ep['summary']['pnl_pts']:+.0f} pts")
        elif time.time() - last_status > 30:
            last_status = time.time()
            board.live_status(feed.label, "aguardando candle novo (mercado fechado?)" if df is None or df.empty or team.closed_day
                              else "aguardando o proximo candle fechar")
        # preco segundo a segundo para as telas (o Emillion continua decidindo so no fechamento do minuto)
        if hasattr(feed, "forming"):
            f = feed.forming()
            if f:
                if broker:
                    f["broker_pos"] = broker.position()
                board.tick(f)
        if feed.finished():
            print("fim do pregao simulado")
            return
        time.sleep(poll)
