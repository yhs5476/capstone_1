@echo off
echo ===================================
echo 음성 파이프라인 관리자 권한 실행
echo ===================================
echo.

REM 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% == 0 (
    echo 관리자 권한으로 실행 중...
    echo.
) else (
    echo 관리자 권한이 필요합니다.
    echo 이 배치 파일을 마우스 우클릭 후 "관리자 권한으로 실행"을 선택해주세요.
    echo.
    pause
    exit /b 1
)

REM Conda 환경 활성화
echo Conda 환경 활성화 중...
call C:\Users\yhs54\anaconda3\Scripts\activate
call conda activate torchcuda311

REM 파이프라인 실행
echo.
echo 음성 파이프라인 실행 중...
echo.
python audio_pipeline.py

echo.
echo 실행 완료!
pause
