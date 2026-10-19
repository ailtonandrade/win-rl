@echo off
rem Treina um agente do zero e mostra no painel ele aprendendo: a cada 20 mil passos
rem a politica atual joga um pregao da validacao e aparece no navegador.
rem Configuracao: scalp (max 15 candles em posicao) + penalidade de 5 pts por entrada.
rem Argumentos extras vao direto para o train.py, ex.: assistir_treino.bat --seed 3 --steps 2000000
cd /d "%~dp0"

start "painel WIN RL" /min .venv\Scripts\python.exe scripts\serve.py
timeout /t 2 /nobreak >nul
start "" http://localhost:8765
.venv\Scripts\python.exe -W ignore scripts\train.py --data WIN$N --steps 1000000 --max-hold 15 --penalty 5 --tag aovivo --watch 20000 %*
echo.
echo Treino terminou. Modelo em models\aovivo_WIN_N_s0 (ou a seed escolhida).
pause
