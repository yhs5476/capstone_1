#!/usr/bin/env python3
"""
다국어 음성 파이프라인 실행 스크립트
"""

import sys
from audio_pipeline import main

def show_language_menu():
    """언어 선택 메뉴 표시"""
    languages = {
        '1': ('ko', '한국어'),
        '2': ('ja', '日本語 (일본어)'),
        '3': ('en', 'English (영어)'),
        '4': ('zh', '中文 (중국어)'),
        '5': ('es', 'Español (스페인어)'),
        '6': ('fr', 'Français (프랑스어)'),
        '7': ('de', 'Deutsch (독일어)'),
        '8': ('ru', 'Русский (러시아어)'),
        '0': (None, '자동 감지')
    }
    
    print("🌐 언어를 선택하세요:")
    print("-" * 30)
    for key, (code, name) in languages.items():
        print(f"{key}. {name}")
    print("-" * 30)
    
    while True:
        choice = input("선택 (0-8): ").strip()
        if choice in languages:
            return languages[choice][0]
        else:
            print("❌ 잘못된 선택입니다. 다시 입력해주세요.")

def main_multilang():
    """다국어 메인 함수"""
    print("=== 다국어 음성 파이프라인 ===")
    print("audio_input → 노이즈제거 → audio_out → STT → script_output")
    print()
    
    # 언어 선택
    target_language = show_language_menu()
    
    print()
    if target_language:
        print(f"✅ 선택된 언어: {target_language}")
    else:
        print("✅ 자동 감지 모드")
    print()
    
    # 파이프라인 실행
    main(target_language=target_language)

if __name__ == "__main__":
    try:
        main_multilang()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)
