"""Teste do envio de ordens (CONTA DEMO): compra 1 contrato do WIN vigente, espera, e zera.
Uso: python scripts/teste_ordem.py"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from winrl.broker import MT5Broker  # noqa: E402
from winrl.team import load_settings  # noqa: E402

b = MT5Broker(load_settings())          # aborta se nao for conta demo
print("broker:", b.label)
print("antes:", b.status())
problem = b.check_trading_enabled()
if problem:
    print("NAO ENVIADO:", problem)
    raise SystemExit(1)
print("compra 1:", b.sync(1, "teste compra"))
time.sleep(3)
print("posicao:", b.position(), "| pnl dia R$", round(b.pnl_today(), 2))
print("zera:", b.sync(0, "teste zera"))
time.sleep(1)
print("depois:", b.status())
