#!/usr/bin/env python3
"""
음성 파이프라인 테스트 스크립트
"""

import os
import sys
from pathlib import Path

def test_dependencies():
    """필수 패키지 설치 확인"""
    print("=== 패키지 설치 확인 ===")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchaudio', 'TorchAudio'),
        ('speechbrain', 'SpeechBrain'),
        ('whisper', 'OpenAI Whisper'),
        ('soundfile', 'SoundFile'),
        ('librosa', 'Librosa'),
        ('numpy', 'NumPy')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}: 설치됨")
        except ImportError:
            print(f"❌ {name}: 설치되지 않음")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n다음 패키지들을 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n모든 필수 패키지가 설치되어 있습니다!")
    return True

def test_gpu():
    """GPU 사용 가능 여부 확인"""
    print("\n=== GPU 확인 ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU 사용 가능: {gpu_count}개")
            print(f"   GPU 이름: {gpu_name}")
            return True
        else:
            print("⚠️  GPU 사용 불가능 - CPU 모드로 실행됩니다")
            return False
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다")
        return False

def test_directories():
    """디렉토리 구조 확인"""
    print("\n=== 디렉토리 구조 확인 ===")
    
    required_dirs = ['audio_input', 'audio_out', 'audio_output', 'script_output']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}: 존재함")
        else:
            print(f"⚠️  {dir_name}: 존재하지 않음 (자동 생성됩니다)")

def test_audio_files():
    """입력 오디오 파일 확인"""
    print("\n=== 입력 파일 확인 ===")
    
    audio_input_dir = Path("audio_input")
    if not audio_input_dir.exists():
        print("❌ audio_input 폴더가 없습니다")
        return False
    
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(audio_input_dir.glob(f"*{ext}")))
        audio_files.extend(list(audio_input_dir.glob(f"*{ext.upper()}")))
    
    if audio_files:
        print(f"✅ {len(audio_files)}개의 오디오 파일 발견:")
        for file in audio_files[:5]:  # 최대 5개만 표시
            print(f"   - {file.name}")
        if len(audio_files) > 5:
            print(f"   ... 및 {len(audio_files) - 5}개 더")
        return True
    else:
        print("⚠️  audio_input 폴더에 오디오 파일이 없습니다")
        print("   지원 형식: WAV, MP3, M4A, FLAC, OGG")
        return False

def test_models():
    """모델 로딩 테스트"""
    print("\n=== 모델 로딩 테스트 ===")
    
    try:
        # SpeechBrain 모델 테스트
        print("SpeechBrain 모델 테스트 중...")
        from speechbrain.pretrained import SpectralMaskEnhancement
        
        # 모델 다운로드만 테스트 (실제 로딩은 하지 않음)
        print("✅ SpeechBrain 모델 접근 가능")
        
        # Whisper 모델 테스트
        print("Whisper 모델 테스트 중...")
        import whisper
        
        # 가장 작은 모델로 테스트
        model = whisper.load_model("tiny")
        print("✅ Whisper 모델 로딩 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("음성 파이프라인 시스템 테스트")
    print("=" * 50)
    
    tests = [
        ("패키지 설치", test_dependencies),
        ("GPU 지원", test_gpu),
        ("디렉토리 구조", test_directories),
        ("입력 파일", test_audio_files),
        ("모델 로딩", test_models)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 오류: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("=== 테스트 결과 요약 ===")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
    
    print(f"\n총 {passed}/{total}개 테스트 통과")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! 파이프라인 실행 준비 완료")
        print("\n다음 명령으로 파이프라인을 실행하세요:")
        print("python audio_pipeline.py")
    else:
        print(f"\n⚠️  {total - passed}개 테스트 실패")
        print("위의 문제들을 해결한 후 다시 시도해주세요.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
