@echo off
REM Запускає pytest через transcription env
set PYTHON=C:\Users\makro\miniconda3\envs\transcription\python.exe
set KMP_DUPLICATE_LIB_OK=TRUE

%PYTHON% -m pytest tests\ -v --tb=short %*
