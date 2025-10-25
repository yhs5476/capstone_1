#!/usr/bin/env python3
"""
음성 파일 노이즈 제거 → Whisper STT → 텍스트 저장 파이프라인
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
    from speechbrain.utils.fetching import LocalStrategy
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
    
    def _load_whisper(self, model_size="base"):
        """Whisper 모델 로드"""
        if self.whisper_model is None:
            try:
                logger.info(f"Whisper 모델 ({model_size}) 로딩 중...")
                self.whisper_model = whisper.load_model(model_size, device=self.device)
                logger.info("Whisper 모델 로딩 완료")
            except Exception as e:
                logger.error(f"Whisper 모델 로딩 실패: {e}")
                raise
    
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
            self._save_simple_transcript(simple_file, result)
            logger.info(f"간단한 전사 파일 생성: {simple_file}")
            
            # SRT 자막 파일 생성 (요청된 경우)
            if srt_file:
                self._save_srt_file(srt_file, result)
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
    
    def _save_srt_file(self, srt_file, result):
        """SRT 자막 파일 생성 (화자 정보 포함)"""
        with open(srt_file, 'w', encoding='utf-8') as f:
            if "segments" in result:
                # 스마트 화자 할당 사용
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
    
    def _save_simple_transcript(self, simple_file, result):
        """간단한 타임스탬프+화자 정보 텍스트 파일 생성"""
        with open(simple_file, 'w', encoding='utf-8') as f:
            if "segments" in result:
                # 스마트 화자 할당
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
        """세그먼트 특성을 기반으로 스마트 화자 할당"""
        if not segments:
            return []
        
        speaker_assignments = []
        current_speaker = 'A'
        last_end_time = 0
        speaker_map = {}
        speaker_count = 0
        
        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", start_time + 1)
            text = segment.get("text", "").strip()
            
            if not text:
                speaker_assignments.append(f"화자{current_speaker}")
                continue
            
            # 화자 변경 조건들
            should_change_speaker = False
            curr_duration = end_time - start_time  # 현재 세그먼트 길이 계산
            
            # 1. 긴 침묵 (2초 이상) 후에는 화자가 바뀔 가능성이 높음
            silence_duration = start_time - last_end_time
            if silence_duration > 2.0:
                should_change_speaker = True
            
            # 2. 문장 길이가 매우 다른 경우 (화자의 말하기 패턴 차이)
            if i > 0:
                prev_duration = segments[i-1].get("end", 0) - segments[i-1].get("start", 0)
                duration_ratio = max(curr_duration, prev_duration) / (min(curr_duration, prev_duration) + 0.1)
                if duration_ratio > 2.5:  # 길이 차이가 2.5배 이상
                    should_change_speaker = True
            
            # 3. 특정 패턴 감지 (질문, 대답, 감탄사 등)
            text_lower = text.lower()
            question_patterns = ['?', '？', 'どう', 'なに', 'なん', '何']
            response_patterns = ['はい', 'そう', 'うん', 'ええ', 'いえ', 'いや']
            
            if i > 0:
                prev_text = segments[i-1].get("text", "").lower()
                # 이전이 질문이고 현재가 대답인 경우
                if any(p in prev_text for p in question_patterns) and any(p in text_lower for p in response_patterns):
                    should_change_speaker = True
            
            # 4. 연속된 짧은 발화 (대화의 특성)
            if curr_duration < 1.5 and silence_duration < 0.5 and i > 0:
                prev_duration = segments[i-1].get("end", 0) - segments[i-1].get("start", 0)
                if prev_duration < 1.5:  # 둘 다 짧은 발화면 화자가 다를 가능성
                    should_change_speaker = True
            
            # 화자 변경 실행
            if should_change_speaker and i > 0:
                # 기존 화자들 중에서 가장 오래 안 나온 화자 선택
                available_speakers = ['A', 'B', 'C']
                if current_speaker in available_speakers:
                    available_speakers.remove(current_speaker)
                
                if available_speakers:
                    # 가장 최근에 사용되지 않은 화자 선택
                    for speaker in available_speakers:
                        speaker_name = f"화자{speaker}"
                        if speaker_name not in [speaker_assignments[j] for j in range(max(0, i-3), i)]:
                            current_speaker = speaker
                            break
                    else:
                        current_speaker = available_speakers[0]
            
            speaker_name = f"화자{current_speaker}"
            speaker_assignments.append(speaker_name)
            last_end_time = end_time
        
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
