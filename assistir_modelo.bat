@echo off
rem Assiste um modelo JA TREINADO (congelado) jogando os pregoes de teste, candle a candle.
rem Uso: assistir_modelo.bat [pasta_do_modelo] [train|val|test]
rem Exemplo: assistir_modelo.bat models\abl-penalty_WIN_N_s0 test
cd /d "%~dp0"
set MODEL=%~1
if "%MODEL%"=="" set MODEL=models\abl-penalty_WIN_N_s2
set SPLIT=%~2
if "%SPLIT%"=="" set SPLIT=test

start "painel WIN RL" /min .venv\Scripts\python.exe scripts\serve.py
timeout /t 2 /nobreak >nul
start "" http://localhost:8765
.venv\Scripts\python.exe -W ignore scripts\watch.py --model %MODEL% --split %SPLIT%
echo.
echo Painel continua aberto em http://localhost:8765 (feche a janela "painel WIN RL" para desligar).
pause
