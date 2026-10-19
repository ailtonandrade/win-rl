@echo off
rem Da a nota de um modelo treinado nos pregoes que ele NUNCA viu (teste),
rem comparando com um robo aleatorio que opera com a mesma frequencia.
rem Uso: avaliar_modelo.bat models\aovivo_WIN_N_s0
cd /d "%~dp0"
set MODEL=%~1
if "%MODEL%"=="" set MODEL=models\aovivo_WIN_N_s0
.venv\Scripts\python.exe -W ignore scripts\evaluate.py %MODEL% --split test
pause
