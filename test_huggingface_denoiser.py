#!/usr/bin/env python3
"""
Hugging Face 오디오 노이즈 제거 테스트 스크립트
"""

import os
import sys

def test_imports():
    """필요한 라이브러리들이 설치되어 있는지 확인"""
    print("=== 라이브러리 설치 확인 ===")
    
    required_packages = [
        'speechbrain',
        'torch',
        'torchaudio', 
        'librosa',
        'soundfile',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 설치됨")
        except ImportError:
            print(f"❌ {package}: 설치되지 않음")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n다음 패키지들을 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n모든 필수 패키지가 설치되어 있습니다!")
    return True

def test_gpu_availability():
    """GPU 사용 가능 여부 확인"""
    print("\n=== GPU 사용 가능성 확인 ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            print(f"✅ GPU 사용 가능: {gpu_count}개")
            print(f"   현재 GPU: {gpu_name}")
            return True
        else:
            print("⚠️  GPU 사용 불가능 - CPU 모드로 실행됩니다")
            return False
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다")
        return False

def test_model_loading():
    """모델 로딩 테스트"""
    print("\n=== 모델 로딩 테스트 ===")
    
    try:
        # 모듈 임포트 시도
        from rnnaudio_huggingface import HuggingFaceDenoiser
        print("✅ 모듈 임포트 성공")
        
        # 모델 초기화 시도 (CPU 모드로)
        print("모델 초기화 중... (시간이 걸릴 수 있습니다)")
        model = HuggingFaceDenoiser(model_type="spectral_mask", use_gpu=False)
        print("✅ 모델 로딩 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {str(e)}")
        return False

def test_demo_execution():
    """데모 실행 테스트"""
    print("\n=== 데모 실행 테스트 ===")
    
    try:
        from rnnaudio_huggingface import main
        print("데모 실행 중...")
        main()
        print("✅ 데모 실행 성공")
        return True
        
    except Exception as e:
        print(f"❌ 데모 실행 실패: {str(e)}")
        return False

def main():
    """메인 테스트 함수"""
    print("Hugging Face 오디오 노이즈 제거 시스템 테스트")
    print("=" * 50)
    
    # 1. 라이브러리 설치 확인
    if not test_imports():
        print("\n테스트 중단: 필수 라이브러리가 설치되지 않았습니다.")
        return
    
    # 2. GPU 확인
    gpu_available = test_gpu_availability()
    
    # 3. 모델 로딩 테스트
    if not test_model_loading():
        print("\n테스트 중단: 모델 로딩에 실패했습니다.")
        print("해결 방법:")
        print("1. 인터넷 연결을 확인하세요")
        print("2. 충분한 디스크 공간이 있는지 확인하세요")
        print("3. speechbrain이 올바르게 설치되었는지 확인하세요")
        return
    
    # 4. 데모 실행 테스트
    if test_demo_execution():
        print("\n🎉 모든 테스트 통과!")
        print("\n시스템이 정상적으로 작동합니다.")
        print("이제 실제 오디오 파일로 노이즈 제거를 시도해보세요.")
    else:
        print("\n⚠️  데모 실행에 문제가 있습니다.")
        print("하지만 기본 설정은 완료되었으므로 수동으로 테스트해보세요.")

if __name__ == "__main__":
    main()
