import MetaTrader5 as mt5, pandas as pd
from datetime import datetime, timedelta

mt5.initialize(); mt5.symbol_select("WIN$N", True)
r = mt5.copy_rates_from_pos("WIN$N", mt5.TIMEFRAME_M1, 0, 100000)
print("from_pos:", None if r is None else len(r), mt5.last_error())
if r is not None and len(r):
    t = pd.to_datetime(r["time"], unit="s"); print(t.min(), "->", t.max())
r = mt5.copy_rates_range("WIN$N", mt5.TIMEFRAME_M1, datetime.now() - timedelta(days=7), datetime.now() + timedelta(days=1))
print("range 7d:", None if r is None else len(r), mt5.last_error())
mt5.shutdown()
