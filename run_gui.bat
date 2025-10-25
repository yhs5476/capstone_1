@echo off
echo ================================
echo   Audio Translator GUI 시작
echo ================================
echo.

REM 가상환경 활성화 (Anaconda)
call C:\Users\yhs54\anaconda3\Scripts\activate.bat torchcuda311

REM Python 스크립트 실행
python audio_translator_gui.py

REM 오류 발생시 대기
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 오류가 발생했습니다. 아무 키나 눌러 종료하세요.
    pause
)
