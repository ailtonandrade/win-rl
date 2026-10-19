"""Mostra como a conta e o contrato vigente do WIN estao configurados para operar. Somente leitura."""
from datetime import datetime

import MetaTrader5 as mt5

if not mt5.initialize():
    raise SystemExit(f"initialize falhou: {mt5.last_error()}")
acc, term = mt5.account_info(), mt5.terminal_info()
print(f"conta={acc.login} servidor={acc.server} modo={'DEMO' if acc.trade_mode == 0 else 'NAO-DEMO'} "
      f"margin_mode={acc.margin_mode} (0=netting 1=exchange 2=hedging) saldo={acc.balance} trade_allowed={acc.trade_allowed} "
      f"trade_expert={acc.trade_expert}")
print(f"terminal: trade_allowed (botao Algo Trading)={term.trade_allowed} tradeapi_disabled={getattr(term, 'tradeapi_disabled', None)}")
now = datetime.now().timestamp()
contracts = sorted((s for s in mt5.symbols_get("WIN*") if s.expiration_time > now and not s.name.startswith("WIN$") and "@" not in s.name),
                   key=lambda s: s.expiration_time)
cur = contracts[0]
mt5.symbol_select(cur.name, True)
info = mt5.symbol_info(cur.name)
tick = mt5.symbol_info_tick(cur.name)
print(f"contrato vigente={info.name} vence={datetime.fromtimestamp(info.expiration_time):%Y-%m-%d} trade_mode={info.trade_mode} "
      f"exec_mode={info.trade_exemode} filling_mode={info.filling_mode} order_mode={info.order_mode} "
      f"vol_min={info.volume_min} step={info.volume_step} tick_size={info.trade_tick_size} stops_level={info.trade_stops_level}")
print(f"tick: bid={tick.bid} ask={tick.ask} last={tick.last} hora_servidor={datetime.utcfromtimestamp(tick.time)}")
print("posicoes abertas:", [(p.symbol, p.type, p.volume, p.magic) for p in (mt5.positions_get() or [])])
print("ordens pendentes:", [(o.symbol, o.type, o.volume_current) for o in (mt5.orders_get() or [])])
mt5.shutdown()
