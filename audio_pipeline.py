#!/usr/bin/env python3
"""
음성 파일 노이즈 제거 → Whisper STT → 음성 특성 기반 화자분리 → 텍스트 저장 파이프라인

필요한 라이브러리:
- librosa: 고급 음성 특성 추출 (pip install librosa)
- scikit-learn: 클러스터링 (pip install scikit-learn)
- numpy: 기본 수치 연산 (pip install numpy)
"""

import os
import sys
import glob
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import torch
import torchaudio
import whisper
import soundfile as sf
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SpeechBrain import를 조건부로 처리
try:
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SPEECHBRAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SpeechBrain 로딩 실패: {e}")
    SPEECHBRAIN_AVAILABLE = False

class AudioPipeline:
    """음성 및 비디오 파이프라인 클래스"""
    
    def __init__(self, use_gpu=True, target_language=None):
        """
        파이프라인 초기화
        
        Args:
            use_gpu (bool): GPU 사용 여부
            target_language (str): 대상 언어 ('ko', 'ja', 'en', None=자동감지)
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.target_language = target_language
        
        # 폴더 경로 설정
        self.audio_input_dir = Path("audio_input")
        self.audio_out_dir = Path("audio_out") 
        self.audio_output_dir = Path("audio_output")
        self.script_output_dir = Path("script_output")
        
        # 폴더 생성
        self._create_directories()
        
        # 모델 초기화
        self.denoiser = None
        self.whisper_model = None
        self.speaker_encoder = None  # ECAPA-VOXCELEB 화자분리 모델
        
        # 지원 언어 정보
        self.supported_languages = {
            'ko': '한국어',
            'ja': '日本語',
            'en': 'English',
            'zh': '中文',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch',
            'ru': 'Русский'
        }
        
        # 지원 파일 형식
        self.audio_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
        self.video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        self.supported_formats = self.audio_formats + self.video_formats
        
        # FFmpeg 사용 가능 여부 확인
        self.ffmpeg_available = self._check_ffmpeg()
        
        logger.info(f"파이프라인 초기화 완료 - 디바이스: {self.device}")
        if target_language:
            lang_name = self.supported_languages.get(target_language, target_language)
            logger.info(f"대상 언어: {lang_name} ({target_language})")
        else:
            logger.info("언어: 자동 감지 모드")
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        for directory in [self.audio_input_dir, self.audio_out_dir, 
                         self.audio_output_dir, self.script_output_dir]:
            directory.mkdir(exist_ok=True)
            logger.info(f"디렉토리 생성/확인: {directory}")
    
    def _check_ffmpeg(self):
        """FFmpeg 설치 여부 확인"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ FFmpeg 사용 가능")
                return True
            else:
                logger.warning("⚠️ FFmpeg 실행 실패")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            logger.warning("⚠️ FFmpeg를 찾을 수 없습니다. 비디오 파일 처리가 제한됩니다.")
            return False
    
    def _is_video_file(self, file_path):
        """비디오 파일인지 확인"""
        return Path(file_path).suffix.lower() in self.video_formats
    
    def _is_audio_file(self, file_path):
        """오디오 파일인지 확인"""
        return Path(file_path).suffix.lower() in self.audio_formats
    
    def extract_audio_from_video(self, video_file, output_audio_file=None):
        """
        FFmpeg를 사용하여 비디오에서 오디오 추출
        
        Args:
            video_file (str): 입력 비디오 파일 경로
            output_audio_file (str, optional): 출력 오디오 파일 경로
            
        Returns:
            str: 추출된 오디오 파일 경로
        """
        if not self.ffmpeg_available:
            raise RuntimeError("FFmpeg를 사용할 수 없습니다. FFmpeg를 설치해주세요.")
        
        video_path = Path(video_file)
        if not video_path.exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_file}")
        
        # 출력 파일명 생성
        if output_audio_file is None:
            output_audio_file = self.audio_input_dir / f"{video_path.stem}_extracted.wav"
        else:
            output_audio_file = Path(output_audio_file)
        
        # 출력 디렉토리 생성
        output_audio_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"🎬 비디오에서 오디오 추출 시작: {video_file}")
            
            # FFmpeg 명령어 구성
            cmd = [
                'ffmpeg',
                '-i', str(video_path),          # 입력 비디오 파일
                '-vn',                          # 비디오 스트림 제거
                '-acodec', 'pcm_s16le',         # 16-bit PCM 인코딩
                '-ar', '16000',                 # 16kHz 샘플링 레이트
                '-ac', '1',                     # 모노 채널
                '-y',                           # 기존 파일 덮어쓰기
                str(output_audio_file)          # 출력 오디오 파일
            ]
            
            # FFmpeg 실행
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ 오디오 추출 완료: {output_audio_file}")
                return str(output_audio_file)
            else:
                error_msg = f"FFmpeg 오디오 추출 실패: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            error_msg = "FFmpeg 처리 시간 초과 (5분)"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"오디오 추출 중 오류 발생: {e}"
            logger.error(error_msg)
            raise
    
    def _load_denoiser(self):
        """노이즈 제거 모델 로드"""
        if self.denoiser is None:
            if not SPEECHBRAIN_AVAILABLE:
                logger.warning("SpeechBrain을 사용할 수 없어 대안 방법 사용")
                self._load_denoiser_alternative()
                return
                
            try:
                logger.info("노이즈 제거 모델 로딩 중...")
                
                # Windows 권한 문제 해결을 위해 LocalStrategy 사용
                import os
                os.environ['SPEECHBRAIN_CACHE_STRATEGY'] = 'LOCAL'
                
                # 절대 경로로 savedir 설정
                savedir = os.path.abspath("pretrained_models/metricgan-plus-voicebank")
                
                self.denoiser = SpectralMaskEnhancement.from_hparams(
                    source="speechbrain/metricgan-plus-voicebank",
                    savedir=savedir,
                    run_opts={"device": self.device}
                )
                logger.info("노이즈 제거 모델 로딩 완료")
            except Exception as e:
                logger.error(f"노이즈 제거 모델 로딩 실패: {e}")
                logger.info("대안 방법으로 재시도...")
                try:
                    self._load_denoiser_alternative()
                except Exception as e2:
                    logger.error(f"대안 방법도 실패: {e2}")
                    raise e
    
    def _load_denoiser_alternative(self):
        """대안 노이즈 제거 방법 (간단한 스펙트럼 필터링)"""
        logger.info("대안 노이즈 제거 방법 사용 (간단한 필터링)")
        self.denoiser = "simple_filter"  # 플래그로 사용
    
    def _simple_denoise(self, waveform, sample_rate):
        """간단한 노이즈 제거 (스펙트럼 필터링)"""
        try:
            # 간단한 고역 통과 필터 적용
            from scipy.signal import butter, filtfilt
            
            # 80Hz 이하 저주파 노이즈 제거
            nyquist = sample_rate / 2
            low_cutoff = 80 / nyquist
            b, a = butter(4, low_cutoff, btype='high')
            
            # 필터 적용
            filtered = filtfilt(b, a, waveform.numpy())
            
            # 정규화
            filtered = filtered / np.max(np.abs(filtered)) * 0.8
            
            return torch.from_numpy(filtered).float()
            
        except ImportError:
            logger.warning("scipy가 설치되지 않아 간단한 정규화만 적용")
            # 간단한 정규화만 적용
            normalized = waveform / torch.max(torch.abs(waveform)) * 0.8
            return normalized
    
    def _load_whisper(self, model_size="large-v3"):
        """Whisper 모델 로드"""
        if self.whisper_model is None:
            try:
                logger.info(f"Whisper 모델 ({model_size}) 로딩 중...")
                self.whisper_model = whisper.load_model(model_size, device=self.device)
                logger.info("Whisper 모델 로딩 완료")
            except Exception as e:
                logger.error(f"Whisper 모델 로딩 실패: {e}")
                raise
    
    def _load_speaker_encoder(self):
        """ECAPA-VOXCELEB 화자분리 모델 로드"""
        if self.speaker_encoder is None:
            if not SPEECHBRAIN_AVAILABLE:
                logger.warning("SpeechBrain을 사용할 수 없어 화자분리 기능을 사용할 수 없습니다")
                return False
                
            try:
                logger.info("ECAPA-VOXCELEB 화자분리 모델 로딩 중...")
                
                # 권한 문제 우회를 위해 임시 디렉토리 사용
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="speechbrain_")
                
                # ECAPA-VOXCELEB 모델 로드
                self.speaker_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=temp_dir,
                    run_opts={"device": self.device}
                )
                logger.info("ECAPA-VOXCELEB 화자분리 모델 로딩 완료")
                return True
            except Exception as e:
                logger.error(f"화자분리 모델 로딩 실패: {e}")
                logger.info("권한 문제일 수 있습니다. 관리자 권한으로 실행하거나 규칙 기반 방식을 사용합니다.")
                self.speaker_encoder = None
                return False
        return True
    
    def denoise_audio(self, input_file, output_file):
        """
        오디오 파일 노이즈 제거
        
        Args:
            input_file (str): 입력 파일 경로
            output_file (str): 출력 파일 경로
        """
        try:
            logger.info(f"노이즈 제거 시작: {input_file}")
            
            # 모델 로드
            self._load_denoiser()
            
            # 오디오 파일 로드
            waveform, sample_rate = torchaudio.load(input_file)
            
            # 모노 채널로 변환 (필요한 경우)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 샘플링 레이트를 16kHz로 변환 (SpeechBrain 모델 요구사항)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # 노이즈 제거 수행
            if self.denoiser == "simple_filter":
                # 대안 방법 사용
                logger.info("간단한 필터링 방법으로 노이즈 제거")
                enhanced_waveform = self._simple_denoise(waveform.squeeze(0), sample_rate)
                enhanced_waveform = enhanced_waveform.unsqueeze(0)
            else:
                # SpeechBrain 모델 사용
                waveform = waveform.to(self.device)
                enhanced_waveform = self.denoiser.enhance_batch(waveform.unsqueeze(0))
                enhanced_waveform = enhanced_waveform.squeeze(0).cpu()
            
            # 출력 파일 저장
            torchaudio.save(output_file, enhanced_waveform, sample_rate)
            
            logger.info(f"노이즈 제거 완료: {output_file}")
            
        except Exception as e:
            logger.error(f"노이즈 제거 실패 ({input_file}): {e}")
            raise
    
    def transcribe_audio(self, audio_file, output_text_file, srt_file=None):
        """
        오디오 파일을 텍스트로 변환 (STT)
        
        Args:
            audio_file (str): 오디오 파일 경로
            output_text_file (str): 출력 텍스트 파일 경로
            srt_file (str, optional): SRT 자막 파일 경로
        """
        try:
            logger.info(f"STT 처리 시작: {audio_file}")
            
            # Whisper 모델 로드
            self._load_whisper()
            
            # 음성 인식 수행 (단어별 타임스탬프 포함)
            transcribe_options = {
                "word_timestamps": True,
                "verbose": True
            }
            
            # 언어 설정
            if self.target_language:
                transcribe_options["language"] = self.target_language
                logger.info(f"지정된 언어로 STT 처리: {self.supported_languages.get(self.target_language, self.target_language)}")
            else:
                logger.info("언어 자동 감지로 STT 처리")
            
            result = self.whisper_model.transcribe(str(audio_file), **transcribe_options)
            
            # 결과 텍스트 추출
            transcribed_text = result["text"].strip()
            
            # 타임스탬프가 포함된 텍스트 파일로 저장
            self._save_transcript_with_timestamps(audio_file, output_text_file, result)
            
            # 간단한 타임스탬프+화자 정보 파일 생성
            simple_file = Path(output_text_file).parent / f"{Path(output_text_file).stem}_simple.txt"
            self._save_simple_transcript(simple_file, result, audio_file)
            logger.info(f"간단한 전사 파일 생성: {simple_file}")
            
            # SRT 자막 파일 생성 (요청된 경우)
            if srt_file:
                self._save_srt_file(srt_file, result, audio_file)
                logger.info(f"SRT 자막 파일 생성: {srt_file}")
            
            logger.info(f"STT 처리 완료: {output_text_file}")
            logger.info(f"인식된 텍스트: {transcribed_text[:100]}...")
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"STT 처리 실패 ({audio_file}): {e}")
            raise
    
    def _save_transcript_with_timestamps(self, audio_file, output_text_file, result):
        """타임스탬프가 포함된 전사 결과 저장"""
        with open(output_text_file, 'w', encoding='utf-8') as f:
            # 헤더 정보
            f.write(f"파일명: {Path(audio_file).name}\n")
            f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"언어: {result.get('language', 'unknown')}\n")
            f.write(f"총 길이: {self._format_time(result.get('duration', 0))}\n")
            f.write("=" * 60 + "\n\n")
            
            # 전체 텍스트
            f.write("📝 전체 텍스트\n")
            f.write("-" * 30 + "\n")
            f.write(result["text"].strip() + "\n\n")
            
            # 세그먼트별 타임스탬프
            f.write("⏰ 타임스탬프별 텍스트\n")
            f.write("-" * 30 + "\n")
            
            if "segments" in result:
                for i, segment in enumerate(result["segments"], 1):
                    start_time = self._format_time(segment["start"])
                    end_time = self._format_time(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"[{start_time} → {end_time}] {text}\n")
                
                # 단어별 타임스탬프 (가능한 경우)
                f.write("\n🔤 단어별 타임스탬프\n")
                f.write("-" * 30 + "\n")
                
                for segment in result["segments"]:
                    if "words" in segment:
                        for word_info in segment["words"]:
                            start_time = self._format_time(word_info["start"])
                            end_time = self._format_time(word_info["end"])
                            word = word_info["word"].strip()
                            confidence = word_info.get("probability", 0)
                            
                            f.write(f"[{start_time}-{end_time}] {word} (신뢰도: {confidence:.2f})\n")
            else:
                f.write("타임스탬프 정보를 사용할 수 없습니다.\n")
    
    def _format_time(self, seconds):
        """초를 MM:SS.mmm 형식으로 변환"""
        if seconds is None:
            return "00:00.000"
        
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"
    
    def _save_srt_file(self, srt_file, result, audio_file=None):
        """SRT 자막 파일 생성 (화자 정보 포함)"""
        with open(srt_file, 'w', encoding='utf-8') as f:
            if "segments" in result:
                # 음성 특성 기반 화자분리 사용
                if audio_file:
                    # 현재 오디오 파일 정보를 임시 저장
                    self._current_audio_file = audio_file
                    speaker_assignments = self._assign_smart_speakers(result["segments"])
                    # 임시 정보 제거
                    delattr(self, '_current_audio_file')
                else:
                    speaker_assignments = self._assign_smart_speakers(result["segments"])
                
                subtitle_index = 1
                for i, segment in enumerate(result["segments"]):
                    start_time = self._format_srt_time(segment["start"])
                    end_time = self._format_srt_time(segment["end"])
                    text = segment["text"].strip()
                    
                    if not text:
                        continue
                    
                    # 할당된 화자 사용
                    speaker_name = speaker_assignments[i]
                    
                    f.write(f"{subtitle_index}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{speaker_name}: {text}\n\n")
                    subtitle_index += 1
            else:
                # 세그먼트 정보가 없는 경우 전체 텍스트를 하나의 자막으로
                duration = result.get("duration", 60)  # 기본 60초
                start_time = self._format_srt_time(0)
                end_time = self._format_srt_time(duration)
                text = result["text"].strip()
                
                f.write("1\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"화자A: {text}\n\n")
    
    def _format_srt_time(self, seconds):
        """초를 SRT 형식 (HH:MM:SS,mmm)으로 변환"""
        if seconds is None:
            return "00:00:00,000"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs % 1) * 1000)
        secs = int(secs)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _save_simple_transcript(self, simple_file, result, audio_file=None):
        """간단한 타임스탬프+화자 정보 텍스트 파일 생성"""
        with open(simple_file, 'w', encoding='utf-8') as f:
            if "segments" in result:
                # 음성 특성 기반 화자분리 사용
                if audio_file:
                    logger.info("음성 특성 기반 화자분리 시작...")
                    # 현재 오디오 파일 정보를 임시 저장
                    self._current_audio_file = audio_file
                    speaker_assignments = self._assign_smart_speakers(result["segments"])
                    # 임시 정보 제거
                    delattr(self, '_current_audio_file')
                else:
                    logger.info("대안 화자 할당 사용...")
                    speaker_assignments = self._assign_smart_speakers(result["segments"])
                
                for i, segment in enumerate(result["segments"]):
                    start_time = self._format_time(segment["start"])
                    text = segment["text"].strip()
                    
                    if not text:
                        continue
                    
                    # 할당된 화자 사용
                    speaker_name = speaker_assignments[i]
                    
                    # [시간] 화자: 텍스트 형식으로 저장
                    f.write(f"[{start_time}] {speaker_name}: {text}\n")
            else:
                # 세그먼트 정보가 없는 경우
                f.write(f"[00:00.000] 화자A: {result['text'].strip()}\n")
    
    def _assign_smart_speakers(self, segments):
        """음성 특성 기반 화자 분리"""
        if not segments:
            return []
        
        logger.info(f"음성 특성 기반 화자 분리 시작 - 총 {len(segments)}개 세그먼트")
        
        # 세그먼트가 너무 적으면 단일 화자
        if len(segments) <= 1:
            logger.info("세그먼트 1개 이하 - 단일 화자로 처리")
            return ["화자A" for _ in segments]
        
        # 오디오 파일에서 음성 특성 추출 시도
        audio_file = getattr(self, '_current_audio_file', None)
        if audio_file:
            return self._voice_feature_based_assignment(audio_file, segments)
        else:
            # 오디오 파일이 없으면 기본 로직
            return self._fallback_speaker_assignment(segments)
    
    def _voice_feature_based_assignment(self, audio_file, segments):
        """음성 특성(주파수, 피치, 스펙트럼) 기반 화자 분리 + 독백 처리"""
        try:
            logger.info("음성 특성 추출 및 독백 분석 중...")
            
            # 독백 여부 사전 판단
            is_monologue = self._detect_monologue_pattern(segments)
            if is_monologue:
                logger.info("독백 패턴 감지 - 독백 전용 처리 모드")
                return self._handle_monologue_segments(segments)
            
            # 오디오 파일 로드
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # 모노 채널로 변환
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 세그먼트별 음성 특성 추출
            voice_features = []
            valid_segments = []
            
            for i, segment in enumerate(segments):
                start_time = segment.get("start", 0)
                end_time = segment.get("end", start_time + 1)
                text = segment.get("text", "").strip()
                
                if not text or end_time <= start_time:
                    continue
                
                # 세그먼트 오디오 추출
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                if start_sample >= waveform.shape[1] or end_sample <= start_sample:
                    continue
                
                segment_audio = waveform[:, start_sample:end_sample]
                
                # 너무 짧은 세그먼트는 건너뛰기 (최소 0.3초)
                if segment_audio.shape[1] < sample_rate * 0.3:
                    continue
                
                # 음성 특성 추출
                features = self._extract_voice_features(segment_audio.squeeze(), sample_rate)
                if features is not None:
                    voice_features.append(features)
                    valid_segments.append((i, segment))
            
            if len(voice_features) < 2:
                logger.info("유효한 음성 특성이 부족 - 단일 화자로 처리")
                return ["화자A" for _ in segments]
            
            # 음성 특성 기반 클러스터링
            speaker_labels = self._cluster_voice_features(voice_features)
            
            # 전체 세그먼트에 화자 할당
            return self._assign_speakers_from_clusters(segments, valid_segments, speaker_labels)
            
        except Exception as e:
            logger.error(f"음성 특성 기반 화자분리 실패: {e}")
            return self._fallback_speaker_assignment(segments)
    
    def _detect_monologue_pattern(self, segments):
        """독백 패턴 감지 (개선된 버전)"""
        try:
            logger.info(f"독백 패턴 분석 중... (세그먼트 수: {len(segments)})")
            
            # 세그먼트가 너무 적어도 독백 가능성 검토
            if len(segments) < 2:
                logger.info("세그먼트 1개 - 독백으로 판단")
                return True
            
            # 1. 침묵 시간 분석
            silence_durations = []
            long_silences = 0  # 2초 이상 침묵 (기준 완화)
            very_long_silences = 0  # 5초 이상 침묵
            
            for i in range(1, len(segments)):
                prev_end = segments[i-1].get("end", 0)
                curr_start = segments[i].get("start", 0)
                silence = curr_start - prev_end
                silence_durations.append(silence)
                
                if silence > 5.0:
                    very_long_silences += 1
                elif silence > 2.0:
                    long_silences += 1
            
            avg_silence = sum(silence_durations) / len(silence_durations) if silence_durations else 0
            
            # 2. 발화 길이 분석
            segment_durations = []
            for segment in segments:
                duration = segment.get("end", 0) - segment.get("start", 0)
                segment_durations.append(duration)
            
            avg_duration = sum(segment_durations) / len(segment_durations)
            max_duration = max(segment_durations) if segment_durations else 0
            
            # 3. 텍스트 패턴 분석
            total_text_length = sum(len(segment.get("text", "")) for segment in segments)
            avg_text_length = total_text_length / len(segments)
            
            # 4. 화자 변경 신호 분석
            speaker_change_signals = 0
            for i in range(1, len(segments)):
                curr_text = segments[i].get("text", "").strip()
                prev_text = segments[i-1].get("text", "").strip()
                
                # 대화 신호 키워드 (질문-응답 패턴)
                question_words = ['?', '？', '뭐', '무엇', 'なに', 'what', 'how']
                response_words = ['네', '예', '아니', 'はい', 'そう', 'yes', 'no']
                
                if (any(q in prev_text for q in question_words) and 
                    any(r in curr_text for r in response_words)):
                    speaker_change_signals += 1
            
            # 5. 독백 판단 기준 (더 관대하게)
            monologue_indicators = [
                very_long_silences == 0,               # 매우 긴 침묵(5초+)이 없음
                avg_silence < 2.0,                     # 평균 침묵이 2초 미만 (완화)
                avg_duration > 1.5 or max_duration > 4.0,  # 평균 1.5초+ 또는 최대 4초+
                avg_text_length > 10,                  # 평균 텍스트가 10자 이상 (완화)
                speaker_change_signals == 0,           # 화자 변경 신호가 없음
                len(segments) <= 5                     # 세그먼트가 5개 이하 (독백은 보통 적음)
            ]
            
            monologue_score = sum(monologue_indicators)
            
            logger.info(f"독백 분석 결과: 점수 {monologue_score}/6 "
                       f"(매우긴침묵: {very_long_silences}, 긴침묵: {long_silences}, "
                       f"평균침묵: {avg_silence:.1f}초, 평균발화: {avg_duration:.1f}초, "
                       f"최대발화: {max_duration:.1f}초, 평균텍스트: {avg_text_length:.1f}자, "
                       f"화자변경신호: {speaker_change_signals})")
            
            # 6개 중 4개 이상 만족하면 독백으로 판단
            is_monologue = monologue_score >= 4
            
            if is_monologue:
                logger.info("🎤 독백으로 판단됨 - 독백 전용 처리 모드 활성화")
            else:
                logger.info("💬 대화로 판단됨 - 음성 특성 기반 화자분리 진행")
            
            return is_monologue
            
        except Exception as e:
            logger.error(f"독백 패턴 감지 실패: {e}")
            # 오류 시 안전하게 독백으로 처리 (단일 화자 우선)
            logger.info("오류로 인해 독백으로 처리")
            return True
    
    def _handle_monologue_segments(self, segments):
        """독백 세그먼트 전용 처리 - 단일 화자 유지"""
        try:
            logger.info("🎤 독백 모드: 단일 화자로 처리")
            
            # 독백은 기본적으로 단일 화자 (화자A)
            speaker_assignments = ["화자A" for _ in segments]
            
            # 독백 내에서도 명확한 주제 전환이 있는 경우에만 화자 분리
            topic_changes = self._detect_strong_topic_changes(segments)
            
            if topic_changes:
                logger.info(f"독백 내 강한 주제 전환 감지: {len(topic_changes)}개 지점")
                current_speaker = 'A'
                
                for i, segment in enumerate(segments):
                    if i in topic_changes:
                        current_speaker = 'B' if current_speaker == 'A' else 'A'
                        logger.info(f"세그먼트 {i}: 강한 주제 전환 → 화자{current_speaker}")
                    
                    speaker_assignments[i] = f"화자{current_speaker}"
            else:
                logger.info("주제 전환 없음 - 완전한 단일 화자 유지")
            
            # 독백 결과 로깅
            from collections import Counter
            speaker_count = Counter(speaker_assignments)
            logger.info(f"🎤 독백 처리 완료: {dict(speaker_count)}")
            
            return speaker_assignments
            
        except Exception as e:
            logger.error(f"독백 세그먼트 처리 실패: {e}")
            return ["화자A" for _ in segments]
    
    def _detect_strong_topic_changes(self, segments):
        """독백 내 강한 주제 전환만 감지 (매우 엄격한 기준)"""
        try:
            topic_changes = []
            
            # 매우 강한 주제 전환 신호만 감지
            strong_transition_keywords = [
                # 한국어 - 명확한 전환
                '그런데 말이야', '아 그리고', '참 그런데', '아 맞다', '그건 그렇고',
                # 일본어 - 명확한 전환  
                'ところで', 'そういえば', 'あ、そうそう', 'それはそうと',
                # 영어 - 명확한 전환
                'by the way', 'speaking of', 'oh and', 'that reminds me'
            ]
            
            for i in range(1, len(segments)):
                curr_text = segments[i].get("text", "").strip().lower()
                
                # 강한 전환 키워드가 있는 경우만
                if any(keyword in curr_text for keyword in strong_transition_keywords):
                    topic_changes.append(i)
                    logger.debug(f"강한 주제 전환 감지: 세그먼트 {i}")
            
            logger.info(f"강한 주제 전환 지점: {len(topic_changes)}개")
            return topic_changes
            
        except Exception as e:
            logger.error(f"강한 주제 전환 감지 실패: {e}")
            return []
    
    def _detect_topic_changes(self, segments):
        """텍스트 기반 주제 변화 감지"""
        try:
            topic_changes = []
            
            # 간단한 키워드 기반 주제 변화 감지
            for i in range(1, len(segments)):
                curr_text = segments[i].get("text", "").strip()
                prev_text = segments[i-1].get("text", "").strip()
                
                # 주제 변화 신호 키워드 (한국어, 일본어, 영어)
                topic_change_keywords = [
                    # 한국어
                    '그런데', '그리고', '또한', '한편', '그래서', '따라서', '결국', '마지막으로',
                    '첫째', '둘째', '셋째', '다음으로', '이제', '그럼', '그러면',
                    # 일본어  
                    'それで', 'そして', 'また', 'しかし', 'でも', 'ところで', 'さて',
                    'まず', '次に', '最後に', '結局', 'つまり', 'だから',
                    # 영어
                    'however', 'but', 'and', 'also', 'then', 'next', 'finally',
                    'first', 'second', 'third', 'so', 'therefore', 'meanwhile'
                ]
                
                # 현재 텍스트에 주제 변화 키워드가 있는지 확인
                curr_lower = curr_text.lower()
                if any(keyword in curr_lower for keyword in topic_change_keywords):
                    topic_changes.append(i)
                    logger.debug(f"주제 변화 감지: 세그먼트 {i} - {curr_text[:30]}...")
                
                # 텍스트 길이 급변 (긴 설명 후 짧은 요약 등)
                if len(prev_text) > 50 and len(curr_text) < 20:
                    topic_changes.append(i)
                    logger.debug(f"텍스트 길이 급변 감지: 세그먼트 {i}")
            
            logger.info(f"주제 변화 지점: {len(topic_changes)}개 - {topic_changes}")
            return topic_changes
            
        except Exception as e:
            logger.error(f"주제 변화 감지 실패: {e}")
            return []
    
    def _extract_voice_features(self, audio_segment, sample_rate):
        """세그먼트에서 음성 특성 추출 (피치, 스펙트럼 중심, MFCC)"""
        try:
            import librosa
            import numpy as np
            
            # numpy 배열로 변환
            if isinstance(audio_segment, torch.Tensor):
                audio_np = audio_segment.numpy()
            else:
                audio_np = audio_segment
            
            # 1. 기본 주파수 (F0) - 피치
            f0 = librosa.yin(audio_np, fmin=50, fmax=400, sr=sample_rate)
            f0_mean = np.nanmean(f0[f0 > 0]) if np.any(f0 > 0) else 150
            f0_std = np.nanstd(f0[f0 > 0]) if np.any(f0 > 0) else 0
            
            # 2. MFCC (Mel-frequency cepstral coefficients) - 음색 특성
            mfcc = librosa.feature.mfcc(y=audio_np, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # 3. 스펙트럴 중심 (Spectral Centroid) - 음성의 밝기
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate)
            sc_mean = np.mean(spectral_centroid)
            sc_std = np.std(spectral_centroid)
            
            # 4. 스펙트럴 대역폭 (Spectral Bandwidth)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_np, sr=sample_rate)
            sb_mean = np.mean(spectral_bandwidth)
            
            # 5. 영교차율 (Zero Crossing Rate) - 음성의 거칠기
            zcr = librosa.feature.zero_crossing_rate(audio_np)
            zcr_mean = np.mean(zcr)
            
            # 특성 벡터 구성
            features = np.concatenate([
                [f0_mean, f0_std],           # 피치 특성 (2차원)
                mfcc_mean[:8],               # MFCC 평균 (8차원)
                [sc_mean, sc_std],           # 스펙트럴 중심 (2차원)
                [sb_mean],                   # 스펙트럴 대역폭 (1차원)
                [zcr_mean]                   # 영교차율 (1차원)
            ])
            
            # NaN 값 처리
            features = np.nan_to_num(features, nan=0.0)
            
            logger.debug(f"음성 특성 추출 완료: F0={f0_mean:.1f}Hz, SC={sc_mean:.1f}Hz")
            return features
            
        except ImportError:
            logger.warning("librosa가 설치되지 않아 간단한 특성만 추출")
            return self._extract_simple_voice_features(audio_segment, sample_rate)
        except Exception as e:
            logger.error(f"음성 특성 추출 실패: {e}")
            return None
    
    def _extract_simple_voice_features(self, audio_segment, sample_rate):
        """librosa 없이 간단한 음성 특성 추출"""
        try:
            import numpy as np
            
            if isinstance(audio_segment, torch.Tensor):
                audio_np = audio_segment.numpy()
            else:
                audio_np = audio_segment
            
            # 1. RMS 에너지 (음량)
            rms_energy = np.sqrt(np.mean(audio_np**2))
            
            # 2. 영교차율 (음성의 거칠기)
            zero_crossings = np.sum(np.diff(np.sign(audio_np)) != 0)
            zcr = zero_crossings / len(audio_np)
            
            # 3. 스펙트럼 분석 (FFT)
            fft = np.fft.fft(audio_np)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(audio_np), 1/sample_rate)[:len(fft)//2]
            
            # 스펙트럴 중심 (가중 평균 주파수)
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                spectral_centroid = 0
            
            # 주요 주파수 (최대 에너지 주파수)
            dominant_freq = freqs[np.argmax(magnitude)] if len(magnitude) > 0 else 0
            
            # 특성 벡터 구성
            features = np.array([
                rms_energy,
                zcr,
                spectral_centroid,
                dominant_freq,
                np.mean(magnitude),
                np.std(magnitude)
            ])
            
            # NaN 값 처리
            features = np.nan_to_num(features, nan=0.0)
            
            logger.debug(f"간단한 음성 특성 추출: 주파수={dominant_freq:.1f}Hz, 에너지={rms_energy:.3f}")
            return features
            
        except Exception as e:
            logger.error(f"간단한 음성 특성 추출 실패: {e}")
            return None
    
    def _cluster_voice_features(self, voice_features):
        """음성 특성 기반 클러스터링"""
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            features_array = np.array(voice_features)
            
            # 특성 정규화
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_array)
            
            # 최적 클러스터 수 결정 - 더 보수적으로 설정
            n_segments = len(voice_features)
            if n_segments <= 2:
                n_clusters = 1  # 단일 화자
            elif n_segments <= 4:
                n_clusters = 2  # 2명 화자
            elif n_segments <= 7:
                n_clusters = min(2, n_segments - 1)  # 최대 2명
            else:
                n_clusters = min(3, n_segments // 2)  # 최대 3명
            
            if n_clusters == 1:
                logger.info("단일 화자로 클러스터링")
                return [0] * len(voice_features)
            
            # K-means 클러스터링 (여러 번 시도해서 최적 결과 선택)
            best_labels = None
            best_inertia = float('inf')
            
            for attempt in range(5):  # 5번 시도
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42+attempt, n_init=10)
                    labels = kmeans.fit_predict(features_normalized)
                    
                    if kmeans.inertia_ < best_inertia:
                        best_inertia = kmeans.inertia_
                        best_labels = labels
                        best_centers = kmeans.cluster_centers_
                except:
                    continue
            
            if best_labels is None:
                logger.warning("클러스터링 실패 - 단일 화자로 처리")
                return [0] * len(voice_features)
            
            logger.info(f"음성 특성 클러스터링 완료: {n_clusters}명 화자 감지 (관성: {best_inertia:.2f})")
            
            # 클러스터 품질 검증
            unique_labels = len(set(best_labels))
            if unique_labels < n_clusters:
                logger.warning(f"일부 클러스터가 비어있음: {unique_labels}/{n_clusters}")
            
            # 클러스터 중심점 정보 로깅
            for i, center in enumerate(best_centers):
                logger.debug(f"화자{chr(65+i)} 특성: F0={center[0]:.1f}, MFCC1={center[2]:.2f}")
            
            return best_labels
            
        except ImportError:
            logger.warning("scikit-learn이 설치되지 않아 간단한 분류 사용")
            return self._simple_voice_clustering(voice_features)
        except Exception as e:
            logger.error(f"음성 특성 클러스터링 실패: {e}")
            return [0] * len(voice_features)  # 모두 같은 화자로 처리
    
    def _simple_voice_clustering(self, voice_features):
        """간단한 음성 특성 기반 분류"""
        try:
            import numpy as np
            
            features_array = np.array(voice_features)
            
            # 첫 번째 특성 (피치 또는 주파수)을 기준으로 분류
            first_feature = features_array[:, 0]
            
            # 중앙값을 기준으로 2그룹 분할
            median_value = np.median(first_feature)
            labels = (first_feature > median_value).astype(int)
            
            logger.info(f"간단한 음성 분류 완료: 기준값={median_value:.2f}")
            return labels
            
        except Exception as e:
            logger.error(f"간단한 음성 분류 실패: {e}")
            return [0] * len(voice_features)
    
    def _assign_speakers_from_clusters(self, segments, valid_segments, speaker_labels):
        """클러스터링 결과를 전체 세그먼트에 할당"""
        speaker_assignments = []
        label_to_speaker = {}
        
        # 라벨을 화자 이름으로 매핑
        unique_labels = sorted(set(speaker_labels))
        for i, label in enumerate(unique_labels):
            speaker_letter = chr(ord('A') + i)
            label_to_speaker[label] = f"화자{speaker_letter}"
        
        # 유효한 세그먼트의 인덱스와 라벨 매핑
        valid_assignments = {}
        for (seg_idx, segment), label in zip(valid_segments, speaker_labels):
            valid_assignments[seg_idx] = label_to_speaker[label]
        
        # 전체 세그먼트에 화자 할당
        current_speaker = "화자A"
        for i, segment in enumerate(segments):
            if i in valid_assignments:
                current_speaker = valid_assignments[i]
            
            speaker_assignments.append(current_speaker)
        
        # 화자 일관성 후처리
        speaker_assignments = self._post_process_voice_consistency(speaker_assignments, valid_segments, speaker_labels)
        
        # 결과 로깅
        from collections import Counter
        speaker_count = Counter(speaker_assignments)
        logger.info(f"음성 특성 기반 화자 분포: {dict(speaker_count)}")
        
        return speaker_assignments
    
    def _post_process_voice_consistency(self, speaker_assignments, valid_segments, speaker_labels):
        """음성 특성 기반 화자 일관성 후처리"""
        try:
            from collections import Counter
            import numpy as np
            
            logger.info("음성 특성 기반 일관성 후처리 시작...")
            
            # 화자별 세그먼트 수 확인
            speaker_counts = Counter(speaker_assignments)
            logger.info(f"후처리 전 화자 분포: {dict(speaker_counts)}")
            
            # 화자분리 품질 확인 - 너무 적극적인 통합 방지
            total_speakers = len(speaker_counts)
            if total_speakers <= 2:
                logger.info(f"화자 수가 적절함 ({total_speakers}명) - 후처리 건너뛰기")
                return speaker_assignments
            
            # 단일 세그먼트만 가진 화자들 찾기 (더 신중하게)
            isolated_speakers = [speaker for speaker, count in speaker_counts.items() if count == 1]
            
            if not isolated_speakers:
                logger.info("고립된 화자 없음 - 후처리 완료")
                return speaker_assignments
            
            logger.info(f"고립된 화자 발견: {isolated_speakers}")
            
            # 고립된 화자 통합 - 더 신중한 조건
            assignments = speaker_assignments.copy()
            
            for isolated_speaker in isolated_speakers:
                isolated_index = speaker_assignments.index(isolated_speaker)
                
                # 앞뒤 화자 확인
                prev_speaker = None
                next_speaker = None
                
                if isolated_index > 0:
                    prev_speaker = assignments[isolated_index - 1]
                if isolated_index < len(assignments) - 1:
                    next_speaker = assignments[isolated_index + 1]
                
                # 통합 대상 결정 - 더 엄격한 조건
                target_speaker = None
                
                # 1. 앞뒤가 정확히 같은 화자이고, 그 화자가 3개 이상 세그먼트를 가진 경우만 통합
                if (prev_speaker and prev_speaker == next_speaker and 
                    speaker_counts[prev_speaker] >= 3):
                    target_speaker = prev_speaker
                    logger.info(f"고립된 화자 {isolated_speaker} → {target_speaker} (앞뒤 동일, 주요화자)")
                
                # 2. 다른 경우는 통합하지 않음 (화자 다양성 보존)
                else:
                    logger.info(f"고립된 화자 {isolated_speaker} 유지 (화자 다양성 보존)")
                
                # 통합 실행
                if target_speaker:
                    assignments[isolated_index] = target_speaker
            
            # 최종 결과 로깅
            final_counts = Counter(assignments)
            logger.info(f"후처리 후 화자 분포: {dict(final_counts)}")
            
            return assignments
            
        except Exception as e:
            logger.error(f"음성 특성 일관성 후처리 실패: {e}")
            return speaker_assignments
    
    def _fallback_speaker_assignment(self, segments):
        """음성 특성 추출 실패 시 대안 로직"""
        logger.info("대안 화자 할당 로직 사용")
        
        # 1.5초 이상 침묵 기준으로 간단 분리
        speaker_assignments = []
        current_speaker = 'A'
        
        for i, segment in enumerate(segments):
            if i > 0:
                prev_end = segments[i-1].get("end", 0)
                curr_start = segment.get("start", 0)
                silence_duration = curr_start - prev_end
                
                if silence_duration > 1.5:
                    current_speaker = 'B' if current_speaker == 'A' else 'A'
            
            speaker_assignments.append(f"화자{current_speaker}")
        
        return speaker_assignments
    
    def _is_single_speaker(self, segments):
        """간단한 단일 화자 판단 로직"""
        if len(segments) <= 2:
            logger.info("세그먼트 2개 이하 - 단일 화자로 판단")
            return True
        
        # 1.5초 이상 침묵이 있는지만 확인
        long_silence_count = 0
        
        for i in range(1, len(segments)):
            prev_end = segments[i-1].get("end", 0)
            curr_start = segments[i].get("start", 0)
            silence_duration = curr_start - prev_end
            
            if silence_duration > 1.5:
                long_silence_count += 1
        
        # 긴 침묵이 없으면 단일 화자
        is_single = long_silence_count == 0
        
        logger.info(f"단일 화자 판단: {'단일' if is_single else '다중'} "
                   f"(1.5초+ 침묵: {long_silence_count}회, 세그먼트: {len(segments)}개)")
        
        return is_single
    
    def _extract_speaker_embeddings(self, audio_file, segments):
        """ECAPA-VOXCELEB를 사용하여 세그먼트별 화자 임베딩 추출"""
        if not self._load_speaker_encoder():
            logger.warning("화자분리 모델을 사용할 수 없어 규칙 기반 방식을 사용합니다")
            return None
        
        try:
            # 오디오 파일 로드
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # 모노 채널로 변환
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 16kHz로 리샘플링 (ECAPA 모델 요구사항)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            embeddings = []
            valid_segments = []
            
            for segment in segments:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", start_time + 1)
                text = segment.get("text", "").strip()
                
                if not text or end_time <= start_time:
                    continue
                
                # 세그먼트 오디오 추출
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                if start_sample >= waveform.shape[1] or end_sample <= start_sample:
                    continue
                
                segment_audio = waveform[:, start_sample:end_sample]
                
                # 너무 짧은 세그먼트는 건너뛰기 (최소 0.5초)
                if segment_audio.shape[1] < sample_rate * 0.5:
                    continue
                
                # 화자 임베딩 추출
                try:
                    embedding = self.speaker_encoder.encode_batch(segment_audio.to(self.device))
                    embeddings.append(embedding.squeeze().cpu().numpy())
                    valid_segments.append(segment)
                except Exception as e:
                    logger.warning(f"세그먼트 임베딩 추출 실패: {e}")
                    continue
            
            if not embeddings:
                logger.warning("유효한 화자 임베딩을 추출할 수 없습니다")
                return None
            
            return np.array(embeddings), valid_segments
            
        except Exception as e:
            logger.error(f"화자 임베딩 추출 실패: {e}")
            return None
    
    def _cluster_speakers(self, embeddings, min_speakers=2, max_speakers=5):
        """화자 임베딩을 클러스터링하여 화자 구분"""
        try:
            # 코사인 유사도 계산
            similarity_matrix = cosine_similarity(embeddings)
            
            # 거리 행렬로 변환 (1 - 유사도)
            distance_matrix = 1 - similarity_matrix
            
            # 최적의 클러스터 수 결정
            best_n_clusters = min_speakers
            best_score = -1
            
            for n_clusters in range(min_speakers, min(max_speakers + 1, len(embeddings) + 1)):
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric='precomputed',
                        linkage='average'
                    )
                    labels = clustering.fit_predict(distance_matrix)
                    
                    # 실루엣 스코어 계산 (간단한 평가)
                    if len(set(labels)) > 1:
                        from sklearn.metrics import silhouette_score
                        score = silhouette_score(distance_matrix, labels, metric='precomputed')
                        if score > best_score:
                            best_score = score
                            best_n_clusters = n_clusters
                except:
                    continue
            
            # 최종 클러스터링
            clustering = AgglomerativeClustering(
                n_clusters=best_n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)
            
            logger.info(f"화자 클러스터링 완료: {best_n_clusters}명의 화자 감지")
            return labels
            
        except Exception as e:
            logger.error(f"화자 클러스터링 실패: {e}")
            return None
    
    def _assign_ecapa_speakers(self, audio_file, segments):
        """ECAPA-VOXCELEB 기반 실제 화자분리"""
        # 화자 임베딩 추출
        embedding_result = self._extract_speaker_embeddings(audio_file, segments)
        
        if embedding_result is None:
            logger.warning("ECAPA 화자분리 실패, 규칙 기반 방식 사용")
            return self._assign_smart_speakers(segments)
        
        embeddings, valid_segments = embedding_result
        
        # 화자 클러스터링
        cluster_labels = self._cluster_speakers(embeddings)
        
        if cluster_labels is None:
            logger.warning("화자 클러스터링 실패, 규칙 기반 방식 사용")
            return self._assign_smart_speakers(segments)
        
        # 클러스터 라벨을 화자 이름으로 변환
        unique_labels = sorted(set(cluster_labels))
        label_to_speaker = {}
        
        for i, label in enumerate(unique_labels):
            speaker_letter = chr(ord('A') + i)
            label_to_speaker[label] = f"화자{speaker_letter}"
        
        # 전체 세그먼트에 화자 할당
        speaker_assignments = []
        valid_idx = 0
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                # 빈 텍스트는 이전 화자 유지
                if speaker_assignments:
                    speaker_assignments.append(speaker_assignments[-1])
                else:
                    speaker_assignments.append("화자A")
                continue
            
            # 유효한 세그먼트인지 확인
            if valid_idx < len(valid_segments) and segment == valid_segments[valid_idx]:
                # 클러스터링 결과 사용
                cluster_label = cluster_labels[valid_idx]
                speaker_name = label_to_speaker[cluster_label]
                speaker_assignments.append(speaker_name)
                valid_idx += 1
            else:
                # 유효하지 않은 세그먼트는 이전 화자 유지
                if speaker_assignments:
                    speaker_assignments.append(speaker_assignments[-1])
                else:
                    speaker_assignments.append("화자A")
        
        return speaker_assignments
    
    def process_single_file(self, input_file):
        """
        단일 파일 전체 파이프라인 처리 (오디오 및 비디오 지원)
        
        Args:
            input_file (str): 입력 오디오 또는 비디오 파일 경로
        """
        try:
            input_path = Path(input_file)
            file_stem = input_path.stem
            
            # 파일 타입 확인 및 전처리
            if self._is_video_file(input_file):
                logger.info(f"🎬 비디오 파일 감지: {input_file}")
                # 비디오에서 오디오 추출
                audio_file = self.extract_audio_from_video(input_file)
                logger.info(f"📄 처리할 오디오 파일: {audio_file}")
            elif self._is_audio_file(input_file):
                logger.info(f"🎵 오디오 파일 감지: {input_file}")
                audio_file = input_file
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {input_path.suffix}")
            
            # 1단계: 노이즈 제거
            denoised_file = self.audio_out_dir / f"{file_stem}_denoised.wav"
            self.denoise_audio(audio_file, str(denoised_file))
            
            # 2단계: audio_output으로 복사 (파이프라인 구조 유지)
            output_audio_file = self.audio_output_dir / f"{file_stem}_denoised.wav"
            import shutil
            shutil.copy2(denoised_file, output_audio_file)
            
            # 3단계: STT 처리
            transcript_file = self.script_output_dir / f"{file_stem}_transcript.txt"
            srt_file = self.script_output_dir / f"{file_stem}_subtitle.srt"
            transcribed_text = self.transcribe_audio(output_audio_file, transcript_file, srt_file)
            
            logger.info(f"파일 처리 완료: {input_file}")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"파일 처리 실패 ({input_file}): {e}")
            raise
    
    def process_all_files(self):
        """audio_input 폴더의 모든 오디오 및 비디오 파일 처리"""
        # 지원하는 파일 확장자 (오디오 + 비디오)
        file_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg', '*.aac',
                          '*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.flv', '*.wmv']
        
        media_files = []
        for ext in file_extensions:
            media_files.extend(glob.glob(str(self.audio_input_dir / ext)))
        
        if not media_files:
            logger.warning(f"audio_input 폴더에 처리할 미디어 파일이 없습니다.")
            logger.info(f"지원 형식: {', '.join(file_extensions)}")
            return
        
        # 파일 타입별 분류
        audio_files = [f for f in media_files if self._is_audio_file(f)]
        video_files = [f for f in media_files if self._is_video_file(f)]
        
        logger.info(f"총 {len(media_files)}개 파일 처리 시작")
        logger.info(f"  - 오디오 파일: {len(audio_files)}개")
        logger.info(f"  - 비디오 파일: {len(video_files)}개")
        
        if video_files and not self.ffmpeg_available:
            logger.warning(f"⚠️ FFmpeg를 사용할 수 없어 비디오 파일 {len(video_files)}개를 건너뜁니다.")
            media_files = audio_files
        
        results = []
        for i, media_file in enumerate(media_files, 1):
            try:
                file_type = "🎬 비디오" if self._is_video_file(media_file) else "🎵 오디오"
                logger.info(f"[{i}/{len(media_files)}] {file_type} 처리 중: {Path(media_file).name}")
                result = self.process_single_file(media_file)
                results.append({
                    'file': Path(media_file).name,
                    'type': 'video' if self._is_video_file(media_file) else 'audio',
                    'status': 'success',
                    'text': result
                })
            except Exception as e:
                logger.error(f"파일 처리 실패: {Path(media_file).name} - {e}")
                results.append({
                    'file': Path(media_file).name,
                    'type': 'video' if self._is_video_file(media_file) else 'audio',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # 처리 결과 요약 저장
        self._save_processing_summary(results)
        
        logger.info("모든 파일 처리 완료")
        return results
    
    def _save_processing_summary(self, results):
        """처리 결과 요약 저장"""
        summary_file = self.script_output_dir / f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 오디오 파이프라인 처리 결과 요약 ===\n")
            f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 파일 수: {len(results)}\n")
            
            success_count = sum(1 for r in results if r['status'] == 'success')
            failed_count = len(results) - success_count
            
            f.write(f"성공: {success_count}개\n")
            f.write(f"실패: {failed_count}개\n")
            f.write("-" * 50 + "\n\n")
            
            for result in results:
                f.write(f"파일: {result['file']}\n")
                f.write(f"상태: {result['status']}\n")
                if result['status'] == 'success':
                    f.write(f"텍스트: {result['text'][:200]}...\n")
                else:
                    f.write(f"오류: {result['error']}\n")
                f.write("-" * 30 + "\n")
        
        logger.info(f"처리 결과 요약 저장: {summary_file}")

def main(target_language=None):
    """
    메인 함수
    
    Args:
        target_language (str): 대상 언어 코드 ('ko', 'ja', 'en', etc.)
    """
    print("=== 음성 및 비디오 파이프라인 시작 ===")
    print("🎵 오디오: audio_input → 노이즈제거 → audio_out → STT → script_output")
    print("🎬 비디오: video_input → 오디오추출 → 노이즈제거 → STT → script_output")
    print()
    
    # 언어 설정 안내
    if target_language:
        supported_languages = {
            'ko': '한국어', 'ja': '日本語', 'en': 'English', 'zh': '中文',
            'es': 'Español', 'fr': 'Français', 'de': 'Deutsch', 'ru': 'Русский'
        }
        lang_name = supported_languages.get(target_language, target_language)
        print(f"🌐 대상 언어: {lang_name} ({target_language})")
    else:
        print("🌐 언어: 자동 감지 모드")
    print()
    
    try:
        # GPU 사용 가능 여부 확인
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  GPU 사용 불가 - CPU 모드로 실행")
        
        # 파이프라인 초기화
        pipeline = AudioPipeline(use_gpu=use_gpu, target_language=target_language)
        
        # 모든 파일 처리
        results = pipeline.process_all_files()
        
        # 결과 출력
        if results:
            success_count = sum(1 for r in results if r['status'] == 'success')
            print(f"\n🎉 처리 완료: {success_count}/{len(results)}개 파일 성공")
        else:
            print("\n⚠️  처리할 파일이 없습니다.")
            print("audio_input 폴더에 오디오 파일을 넣어주세요.")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}")
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# 사용 예시:
# 
# 1. 오디오 파일 처리:
#    audio_input/ 폴더에 WAV, MP3, M4A, FLAC, OGG 파일 저장 후 실행
#
# 2. 비디오 파일 처리 (FFmpeg 필요):
#    audio_input/ 폴더에 MP4, AVI, MOV, MKV, WEBM 파일 저장 후 실행
#    
# 3. 특정 언어 지정:
#    pipeline = AudioPipeline(target_language='ko')  # 한국어
#    pipeline = AudioPipeline(target_language='ja')  # 일본어
#
# 4. 단일 파일 처리:
#    pipeline = AudioPipeline()
#    result = pipeline.process_single_file("video.mp4")
#
# 5. FFmpeg 설치 확인:
#    pipeline = AudioPipeline()
#    print(f"FFmpeg 사용 가능: {pipeline.ffmpeg_available}")
#
# 출력 파일:
# - audio_out/: 노이즈 제거된 오디오
# - script_output/: 전사 텍스트 및 SRT 자막
# - processing_summary_*.txt: 배치 처리 결과 요약
