@echo off
REM Запускає будь-яку команду через Python з transcription conda env (GPU torch)
REM Використання:
REM   run.bat src\ui_app.py
REM   run.bat -m pytest tests\ -v

set PYTHON=C:\Users\makro\miniconda3\envs\transcription\python.exe
set KMP_DUPLICATE_LIB_OK=TRUE

%PYTHON% %*
