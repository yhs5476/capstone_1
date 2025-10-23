import librosa
import librosa.display
import numpy as np
import soundfile as sf
import os
from speechbrain.inference.enhancement import SpectralMaskEnhancement
from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
import warnings
import subprocess
import sys

# Hugging Face 로그인 (선택사항)
# 환경변수나 별도 설정 파일에서 토큰을 읽어오세요
# import huggingface_hub
# huggingface_hub.login("YOUR_HF_TOKEN_HERE")  # 토큰을 직접 코드에 넣지 마세요!

# --- 1. Hugging Face SpeechBrain 모델 설정 ---
# SpeechBrain의 사전 훈련된 모델을 사용하여 노이즈 제거
# 두 가지 모델 옵션:
# 1. SpectralMaskEnhancement - 일반적인 노이즈 제거
# 2. SepformerSeparation - 더 고급 분리 및 향상

def check_admin_rights():
    """관리자 권한 확인"""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def enable_symlink_privilege():
    """심볼릭 링크 권한 활성화 시도"""
    try:
        import ctypes
        from ctypes import wintypes
        
        # SeCreateSymbolicLinkPrivilege 활성화 시도
        TOKEN_ADJUST_PRIVILEGES = 0x0020
        TOKEN_QUERY = 0x0008
        SE_PRIVILEGE_ENABLED = 0x00000002
        
        hToken = wintypes.HANDLE()
        hProcess = ctypes.windll.kernel32.GetCurrentProcess()
        
        if ctypes.windll.advapi32.OpenProcessToken(hProcess, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ctypes.byref(hToken)):
            return True
    except:
        pass
    return False

class HuggingFaceDenoiser:
    def __init__(self, model_type="spectral_mask", use_gpu=False):
        """
        Hugging Face SpeechBrain 모델을 초기화합니다.
        
        Args:
            model_type (str): "spectral_mask" 또는 "sepformer"
                - spectral_mask: VoiceBank-DEMAND 데이터셋으로 훈련된 일반적인 노이즈 제거 모델
                - sepformer: WHAM! 데이터셋으로 훈련된 고급 음성 분리 모델
            use_gpu (bool): GPU 사용 여부 (CUDA 가속)
        """
        # 모델 타입 저장 (허깅페이스에서 다운로드할 모델 결정)
        self.model_type = model_type
        
        # 디바이스 설정: GPU 사용 시 "cuda", CPU 사용 시 "cpu"
        self.device = "cuda" if use_gpu else "cpu"
        
        # 모델 객체 초기화 (로딩 실패 시 None으로 유지)
        self.model = None
        
        # Windows 권한 문제 해결 시도 (심볼릭 링크 생성 권한 필요)
        print("권한 확인 중...")
        if not check_admin_rights():
            print("⚠️  관리자 권한이 없습니다. 대안 방법을 시도합니다...")
        
        # 허깅페이스 모델 로딩 시도
        try:
            self._load_model()  # 실제 모델 다운로드 및 로딩
        except Exception as e:
            print(f"모델 로딩 실패: {str(e)}")
            print("기본 오디오 처리 방법을 사용합니다.")
            self.model = None  # 로딩 실패 시 fallback 처리 준비
    
    def _load_model(self):
        """
        허깅페이스에서 사전 훈련된 모델을 다운로드하고 로딩합니다.
        권한 문제를 우회하기 위해 사용자 홈 디렉토리를 사용합니다.
        """
        import tempfile
        import os
        
        # 사용자 홈 디렉토리에 모델 캐시 저장 (~/.speechbrain_models/)
        # 시스템 디렉토리 권한 문제를 피하기 위한 방법
        home_dir = os.path.expanduser("~")  # Windows: C:\Users\사용자명
        model_dir = os.path.join(home_dir, "speechbrain_models", self.model_type)
        
        # 모델 저장 디렉토리 생성 (없으면 자동 생성)
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model_type == "spectral_mask":
            # === VoiceBank-DEMAND 데이터셋으로 훈련된 SpectralMask 모델 ===
            # 환경 변수로 SpeechBrain 캐시 경로 지정 (권한 문제 우회)
            os.environ['SPEECHBRAIN_CACHE'] = model_dir
            
            # 허깅페이스에서 모델 다운로드 및 로딩
            # source: 허깅페이스 모델 ID (speechbrain/mtl-mimic-voicebank)
            # savedir: 로컬 저장 경로
            # run_opts: 실행 옵션 (GPU/CPU 설정)
            self.model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/mtl-mimic-voicebank",  # VoiceBank-DEMAND 훈련 모델
                savedir=model_dir,                         # 로컬 캐시 경로
                run_opts={"device": self.device}           # 디바이스 설정
            )
            
        elif self.model_type == "sepformer":
            # === WHAM! 데이터셋으로 훈련된 SepFormer 모델 ===
            os.environ['SPEECHBRAIN_CACHE'] = model_dir
            run_opts = {"device": self.device}
            
            # SepFormer: 더 고급 음성 분리 및 향상 모델
            # WHAM! (WSJ0 Hipster Ambient Mixtures) 데이터셋으로 훈련됨
            self.model = separator.from_hparams(
                source="speechbrain/sepformer-wham-enhancement",  # WHAM! 훈련 모델
                savedir=model_dir,                                # 로컬 캐시 경로
                run_opts=run_opts                                 # 디바이스 설정
            )
        else:
            raise ValueError("model_type은 'spectral_mask' 또는 'sepformer'여야 합니다.")
    
    def enhance_audio(self, input_path, output_path):
        """
        오디오 파일의 노이즈를 제거합니다.
        
        Args:
            input_path (str): 입력 오디오 파일 경로
            output_path (str): 출력 오디오 파일 경로
        """
        if self.model is None:
            print("⚠️  Hugging Face 모델을 사용할 수 없습니다. 기본 처리 방법을 사용합니다.")
            return self._fallback_processing(input_path, output_path)
        
        try:
            if self.model_type == "spectral_mask":
                # SpectralMaskEnhancement 사용
                enhanced_audio = self.model.enhance_file(input_path)
                # 결과를 파일로 저장
                torchaudio.save(output_path, enhanced_audio.cpu(), 16000)
            
            elif self.model_type == "sepformer":
                # SepFormer 사용 (8kHz 샘플링)
                est_sources = self.model.separate_file(path=input_path)
                # 첫 번째 소스 (향상된 음성)를 저장
                torchaudio.save(output_path, est_sources[:, :, 0].detach().cpu(), 8000)
            
            print(f"✅ 노이즈 제거 완료. 결과 파일: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"⚠️  Hugging Face 모델 처리 실패: {str(e)}")
            print("기본 처리 방법으로 전환합니다...")
            return self._fallback_processing(input_path, output_path)
    
    def _fallback_processing(self, input_path, output_path):
        """기본 오디오 처리 (권한 문제 시 대안)"""
        print("🔧 기본 스펙트럴 서브트랙션을 사용합니다...")
        
        # librosa로 기본 노이즈 감소
        y, sr = librosa.load(input_path, sr=16000)
        
        # 스펙트로그램 분석
        D = librosa.stft(y)
        magnitude, phase = librosa.magphase(D)
        
        # 간단한 스펙트럴 서브트랙션
        magnitude_db = librosa.amplitude_to_db(magnitude)
        noise_floor = np.percentile(magnitude_db, 15)  # 하위 15%를 노이즈로 간주
        
        # 적응적 마스킹
        mask = magnitude_db > (noise_floor + 8)  # 8dB 마진
        
        # 부드러운 마스킹 적용
        smooth_mask = np.maximum(mask * 1.0, 0.1)  # 최소 10% 신호 보존
        cleaned_magnitude = magnitude * smooth_mask
        
        # 역변환
        cleaned_stft = cleaned_magnitude * phase
        y_cleaned = librosa.istft(cleaned_stft)
        
        # 정규화
        y_cleaned = y_cleaned / np.max(np.abs(y_cleaned)) * 0.95
        
        # 저장
        sf.write(output_path, y_cleaned, sr)
        print(f"✅ 기본 노이즈 감소 완료. 결과 파일: {output_path}")
        return output_path

# --- 2. 설정 및 모델 초기화 ---
SR = 16000          # 기본 샘플링 레이트 (Hz)

# GPU 사용 가능 여부 확인
try:
    import torch
    use_gpu = torch.cuda.is_available()
    print(f"GPU 사용 가능: {use_gpu}")
except ImportError:
    use_gpu = False
    print("PyTorch가 설치되지 않았습니다. CPU 모드로 실행합니다.")

# 모델 타입 선택: "spectral_mask" 또는 "sepformer"
# spectral_mask: 16kHz, 일반적인 노이즈 제거
# sepformer: 8kHz, 더 고급 분리 및 향상
MODEL_TYPE = "spectral_mask"  # 기본값

# 모델 인스턴스 생성
print(f"Hugging Face {MODEL_TYPE} 모델을 로드하는 중...")
denoiser_model = HuggingFaceDenoiser(model_type=MODEL_TYPE, use_gpu=use_gpu)
print("모델 로드 완료!")

# --- 3. 음성 잡음 제거 함수 (Hugging Face 모델 사용) ---
def apply_denoising_huggingface(noisy_audio_path, output_audio_path, model):
    """
    Hugging Face SpeechBrain 모델을 사용하여 오디오 노이즈를 제거합니다.
    
    Args:
        noisy_audio_path (str): 노이즈가 있는 입력 오디오 파일 경로
        output_audio_path (str): 노이즈 제거된 출력 오디오 파일 경로
        model (HuggingFaceDenoiser): 초기화된 Hugging Face 모델
    
    Returns:
        str: 출력 파일 경로
    """
    try:
        # Hugging Face 모델을 사용하여 노이즈 제거
        result_path = model.enhance_audio(noisy_audio_path, output_audio_path)
        return result_path
    
    except Exception as e:
        print(f"노이즈 제거 중 오류 발생: {str(e)}")
        print("대체 방법으로 librosa를 사용하여 기본 처리를 수행합니다.")
        
        # 대체 방법: 기본 오디오 처리
        y, sr = librosa.load(noisy_audio_path, sr=SR)
        
        # 간단한 스펙트럴 게이팅 적용 (기본 노이즈 감소)
        # 이는 완전한 대체는 아니지만 기본적인 처리를 제공합니다
        D = librosa.stft(y)
        magnitude, phase = librosa.magphase(D)
        
        # 간단한 스펙트럴 서브트랙션
        # 낮은 에너지 부분을 감소시킵니다
        magnitude_db = librosa.amplitude_to_db(magnitude)
        noise_floor = np.percentile(magnitude_db, 20)  # 하위 20%를 노이즈로 간주
        
        # 노이즈 플로어 이하의 신호를 감소
        mask = magnitude_db > (noise_floor + 6)  # 6dB 마진
        cleaned_magnitude = magnitude * mask
        
        # 역변환
        cleaned_stft = cleaned_magnitude * phase
        y_cleaned = librosa.istft(cleaned_stft)
        
        # 저장
        sf.write(output_audio_path, y_cleaned, sr)
        print(f"기본 노이즈 감소 완료. 결과 파일: {output_audio_path}")
        return output_audio_path

# --- 4. 실행 예시 및 사용법 ---

def create_demo_audio():
    """
    데모용 노이즈가 있는 오디오 파일을 생성합니다.
    """
    dummy_noisy_path = "noisy_speech.wav"
    
    if not os.path.exists(dummy_noisy_path):
        # 더 현실적인 테스트 신호 생성
        duration = 3  # 3초
        t = np.linspace(0, duration, SR * duration)
        
        # 기본 음성 신호 (사인파 조합으로 음성 유사 신호 생성)
        speech_like = (
            np.sin(2 * np.pi * 200 * t) * np.exp(-t * 0.5) +  # 기본 주파수
            0.3 * np.sin(2 * np.pi * 400 * t) * np.exp(-t * 0.3) +  # 하모닉
            0.1 * np.sin(2 * np.pi * 800 * t) * np.exp(-t * 0.2)   # 고주파 성분
        )
        
        # 노이즈 추가
        noise = np.random.randn(len(speech_like)) * 0.3
        
        # 신호와 노이즈 결합
        noisy_signal = speech_like + noise
        
        # 정규화
        noisy_signal = noisy_signal / np.max(np.abs(noisy_signal)) * 0.8
        
        sf.write(dummy_noisy_path, noisy_signal, SR)
        print(f"데모용 노이즈 파일 생성: {dummy_noisy_path}")
    
    return dummy_noisy_path

def main():
    """
    메인 실행 함수
    """
    print("=== Hugging Face 오디오 노이즈 제거 데모 ===")
    
    # 실제 오디오 파일 사용 (경로를 수정하세요)
    input_file = "test.mp3"  # 실제 파일 경로로 변경
    # input_file = create_demo_audio()  # 데모용 (실제 파일 사용 시 위 줄 주석 해제하고 이 줄 주석 처리)
    output_file = "denoised_speech_hf.wav"
    
    try:
        # Hugging Face 모델로 노이즈 제거 실행
        print("\n노이즈 제거를 시작합니다...")
        result_path = apply_denoising_huggingface(input_file, output_file, denoiser_model)
        
        print(f"\n✅ 성공! 결과 파일: {result_path}")
        print("\n사용법:")
        print("1. 실제 오디오 파일을 사용하려면 input_file 경로를 변경하세요")
        print("2. 다른 모델을 사용하려면 MODEL_TYPE을 'sepformer'로 변경하세요")
        print("3. GPU를 사용하려면 CUDA가 설치되어 있는지 확인하세요")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print("\n해결 방법:")
        print("1. speechbrain 라이브러리가 설치되어 있는지 확인하세요: pip install speechbrain")
        print("2. 인터넷 연결을 확인하세요 (모델 다운로드 필요)")
        print("3. 충분한 디스크 공간이 있는지 확인하세요")

# 스크립트로 직접 실행할 때만 데모 실행
if __name__ == "__main__":
    main()
else:
    print("모듈이 임포트되었습니다. main() 함수를 호출하여 데모를 실행하세요.")
    print("또는 apply_denoising_huggingface() 함수를 직접 사용하세요.")

# ----------------------------------------------------
# 추가 정보:
# - speechbrain 설치: pip install speechbrain
# - 지원되는 오디오 형식: wav, flac, mp3 등
# - 권장 샘플링 레이트: 16kHz (spectral_mask) 또는 8kHz (sepformer)
# ----------------------------------------------------
