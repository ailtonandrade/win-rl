@echo off
rem EMILLION: edite o megazord.ini (MODO, CONECTAR_MT5...) e rode este arquivo.
rem O painel abre sozinho no navegador. Ctrl+C nesta janela para parar.
cd /d "%~dp0"
title EMILLION
.venv\Scripts\python.exe -W ignore scripts\run.py
pause
