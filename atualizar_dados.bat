@echo off
rem Baixa do MT5 (aberto e logado na conta DEMO) o historico M1 e M5 do WIN$N.
rem So le dados: nao envia nenhuma ordem. Rode antes de treinar para pegar os pregoes novos.
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\export_rates.py WIN$N M1 M5
pause
