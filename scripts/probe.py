"""Conecta no MT5 aberto e mostra conta + simbolos WIN. Somente leitura."""
import MetaTrader5 as mt5

if not mt5.initialize():
    raise SystemExit(f"initialize falhou: {mt5.last_error()}")

acc = mt5.account_info()
term = mt5.terminal_info()
modes = {0: "DEMO", 1: "CONTEST", 2: "REAL"}
print(f"conta={acc.login} servidor={acc.server} corretora={acc.company} modo={modes.get(acc.trade_mode)} moeda={acc.currency}")
print(f"terminal={term.name} build={mt5.version()} conectado={term.connected} maxbars={term.maxbars}")

syms = [s for s in (mt5.symbols_get("*WIN*") or [])]
print(f"\n{len(syms)} simbolos com WIN:")
for s in syms:
    print(f"  {s.name:12} visivel={s.visible} point={s.point} tick_size={s.trade_tick_size} "
          f"tick_value={s.trade_tick_value} spread={s.spread} vol_min={s.volume_min} "
          f"exp={s.expiration_time} path={s.path}")
mt5.shutdown()
