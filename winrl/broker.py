"""Envio de ordens do Emillion para o MT5. Por padrao SO FUNCIONA EM CONTA DEMO.

Conta real so com as duas travas: PERMITIR_CONTA_REAL = true no megazord.ini E digitar o
numero da conta na janela ao iniciar (cada vez que o programa abre).

O Emillion decide uma posicao-alvo (-1, 0, +1 contrato). O broker compara com a posicao
real no MT5 (so as ordens com o MAGIC do Emillion) e manda uma ordem a mercado com a
diferenca. Conta netting (padrao B3): vender 2 com 1 comprado vira 1 vendido.

Travas:
- recusa conta que nao seja DEMO, a nao ser com PERMITIR_CONTA_REAL + confirmacao digitada;
- STOP de emergencia no servidor em cada entrada (se o programa cair, o MT5 fecha sozinho);
- perda maxima no dia (R$) e numero maximo de ordens no dia: atingiu, zera e para de entrar;
- arquivo PARAR (sem extensao) na pasta do projeto: zera e para de entrar enquanto existir;
- cada ordem vai para logs/ordens_AAAA-MM-DD.csv.
"""
import csv
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAGIC = 20261019
KILL_FILE = ROOT / "PARAR"


class BrokerError(SystemExit):
    pass


def confirm_real_account(acc, s):
    """Segunda trava da conta real: so passa se PERMITIR_CONTA_REAL = true e o numero da conta for digitado."""
    if not getattr(s, "permitir_conta_real", False):
        raise BrokerError("CONTA NAO E DEMO. Por padrao o Emillion so envia ordens em conta demo. Nada foi enviado.\n"
                          "(Para conta real: PERMITIR_CONTA_REAL = true no megazord.ini e confirmar ao iniciar.)")
    print("\n" + "!" * 70)
    print("  ATENCAO: CONTA REAL. As ordens do Emillion vao usar DINHEIRO DE VERDADE.")
    print(f"  conta {acc.login} | servidor {acc.server} | {acc.company} | saldo {acc.balance:,.2f} {acc.currency}")
    print(f"  travas: perda max R$ {s.perda_max_dia_rs:.0f}/dia, {s.max_ordens_dia} ordens/dia, stop {s.stop_emergencia_pts:.0f} pts, 1 contrato")
    print("!" * 70)
    try:
        typed = input("Para confirmar, digite o NUMERO DA CONTA (qualquer outra coisa cancela): ").strip()
    except EOFError:
        typed = ""
    if typed != str(acc.login):
        raise BrokerError("Confirmacao nao bate. Nenhuma ordem sera enviada.")
    print("Conta real confirmada.\n")


class MT5Broker:
    def __init__(self, s, prefix="WIN"):
        import MetaTrader5 as mt5
        self.mt5, self.s = mt5, s
        if not mt5.initialize():
            raise BrokerError(f"Nao conectou no MT5: {mt5.last_error()}")
        acc = mt5.account_info()
        if acc is None:
            raise BrokerError("Nao foi possivel ler a conta do MT5. Nada foi enviado.")
        self.real = acc.trade_mode != 0
        if self.real:
            confirm_real_account(acc, s)
        self.netting = acc.margin_mode != 2
        now = datetime.now().timestamp()
        contracts = sorted((x for x in mt5.symbols_get(f"{prefix}*") if x.expiration_time > now and "$" not in x.name and "@" not in x.name),
                           key=lambda x: x.expiration_time)
        if not contracts:
            raise BrokerError(f"Nenhum contrato {prefix} vigente encontrado no MT5")
        self.symbol = contracts[0].name
        mt5.symbol_select(self.symbol, True)
        info = mt5.symbol_info(self.symbol)
        self.tick_size = info.trade_tick_size or 5.0
        self.filling = mt5.ORDER_FILLING_FOK if info.filling_mode & 1 else mt5.ORDER_FILLING_IOC if info.filling_mode & 2 else mt5.ORDER_FILLING_RETURN
        self.orders_today, self.day, self.blocked = 0, None, None
        self.last = None
        self.label = (f"{self.symbol} na CONTA REAL {acc.login}" if self.real else f"{self.symbol} na conta demo {acc.login}") \
            + f" ({'netting' if self.netting else 'hedging'})"

    # ------------------------------------------------------------------ estado
    def check_trading_enabled(self):
        term = self.mt5.terminal_info()
        if not term.trade_allowed:
            return "o botao 'Algo Trading' do MT5 esta DESLIGADO (ligue na barra de ferramentas)"
        return None

    def position(self) -> int:
        poss = [p for p in (self.mt5.positions_get(symbol=self.symbol) or []) if p.magic == MAGIC]
        net = 0.0
        for p in poss:
            net += p.volume if p.type == self.mt5.POSITION_TYPE_BUY else -p.volume
        return int(round(net))

    def pnl_today(self) -> float:
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        deals = self.mt5.history_deals_get(start, datetime.now()) or []
        realized = sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0.0) for d in deals if d.magic == MAGIC)
        floating = sum(p.profit for p in (self.mt5.positions_get(symbol=self.symbol) or []) if p.magic == MAGIC)
        return realized + floating

    def status(self) -> dict:
        pos = self.position()
        return {"symbol": self.symbol, "real": self.real, "position": pos, "pnl_brl": round(self.pnl_today(), 2),
                "orders_today": self.orders_today, "blocked": self.blocked, "last": self.last,
                "algo_trading": self.check_trading_enabled() is None}

    # ------------------------------------------------------------------ ordens
    def _new_day(self):
        today = datetime.now().date()
        if today != self.day:
            self.day, self.orders_today, self.blocked = today, 0, None

    def _guards(self):
        if KILL_FILE.exists():
            return "arquivo PARAR encontrado"
        if self.pnl_today() <= -abs(self.s.perda_max_dia_rs):
            return f"perda maxima do dia atingida (R$ {self.s.perda_max_dia_rs:.0f})"
        if self.orders_today >= self.s.max_ordens_dia:
            return f"maximo de {self.s.max_ordens_dia} ordens no dia atingido"
        return None

    def sync(self, target: int, reason: str) -> dict | None:
        """Leva a posicao real ate `target` (-1, 0, +1). Devolve o resultado da ordem (ou None se nada a fazer)."""
        self._new_day()
        why = self._guards()
        if why:
            if not self.blocked:
                self.blocked = why
                print(f"!! TRAVA: {why}. Zerando e parando de entrar.")
            target = 0
        target = max(-1, min(1, int(target)))
        cur = self.position()
        delta = target - cur
        if delta == 0:
            return None
        problem = self.check_trading_enabled()
        if problem:
            self.last = {"time": datetime.now().strftime("%H:%M:%S"), "ok": False, "msg": problem}
            return self.last
        return self._send(delta, target, reason)

    def flatten(self, reason="zerar"):
        return self.sync(0, reason) if self.position() != 0 else None

    def _send(self, delta: int, target: int, reason: str) -> dict:
        mt5 = self.mt5
        tick = mt5.symbol_info_tick(self.symbol)
        buy = delta > 0
        price = tick.ask if buy else tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": float(abs(delta)),
            "type": mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL, "price": price,
            "deviation": int(self.s.desvio_max_pts), "magic": MAGIC, "comment": f"emillion {reason}"[:31],
            "type_time": mt5.ORDER_TIME_DAY, "type_filling": self.filling,
        }
        opening = target != 0
        if opening and self.s.stop_emergencia_pts > 0:
            gap = round(self.s.stop_emergencia_pts / self.tick_size) * self.tick_size
            req["sl"] = price - gap if target > 0 else price + gap
        res = mt5.order_send(req)
        if res is not None and res.retcode != mt5.TRADE_RETCODE_DONE and "sl" in req:
            # algumas corretoras recusam SL junto da ordem a mercado: manda sem e coloca o stop depois
            req.pop("sl")
            res = mt5.order_send(req)
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                self._attach_stop(target, res.price or price)
        ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        self.orders_today += 1 if ok else 0
        self.last = {"time": datetime.now().strftime("%H:%M:%S"), "ok": ok, "side": "COMPRA" if buy else "VENDA",
                     "volume": abs(delta), "price": res.price if ok else price, "target": target, "reason": reason,
                     "msg": "executada" if ok else f"recusada ({res.retcode if res else mt5.last_error()}: {res.comment if res else ''})"}
        self._log(req, res)
        print(f">> ORDEM {self.last['side']} {abs(delta)} {self.symbol} @ {self.last['price']:,.0f} -> alvo {target:+d} ({reason}): {self.last['msg']}")
        return self.last

    def _attach_stop(self, target, entry):
        mt5 = self.mt5
        gap = round(self.s.stop_emergencia_pts / self.tick_size) * self.tick_size
        for p in mt5.positions_get(symbol=self.symbol) or []:
            if p.magic == MAGIC:
                mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": self.symbol, "position": p.ticket,
                                "sl": entry - gap if target > 0 else entry + gap, "tp": 0.0})

    def _log(self, req, res):
        f = ROOT / "logs" / f"ordens_{datetime.now():%Y-%m-%d}.csv"
        f.parent.mkdir(exist_ok=True)
        new = not f.exists()
        with open(f, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=";")
            if new:
                w.writerow(["hora", "simbolo", "lado", "volume", "preco_pedido", "preco_executado", "sl", "retcode", "comentario", "motivo"])
            w.writerow([datetime.now().strftime("%H:%M:%S"), req["symbol"], "COMPRA" if req["type"] == self.mt5.ORDER_TYPE_BUY else "VENDA",
                        req["volume"], req["price"], getattr(res, "price", ""), req.get("sl", ""), getattr(res, "retcode", ""),
                        getattr(res, "comment", ""), req["comment"]])
