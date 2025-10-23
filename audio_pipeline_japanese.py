#!/usr/bin/env python3
"""
일본어 음성 파이프라인 실행 스크립트
"""

import sys
from audio_pipeline import main

if __name__ == "__main__":
    print("🇯🇵 일본어 음성 파이프라인")
    print("=" * 40)
    
    # 일본어로 설정하여 실행
    main(target_language="ja")
