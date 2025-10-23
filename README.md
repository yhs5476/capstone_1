# 음성 파이프라인 (Audio Pipeline)

음성 파일의 노이즈를 제거하고 Whisper를 사용하여 음성을 텍스트로 변환하는 파이프라인입니다.

## 파이프라인 구조

```
audio_input → 노이즈 제거 → audio_out → STT 처리 → script_output
```

1. **audio_input**: 원본 음성 파일 입력 폴더
2. **audio_out**: 노이즈 제거된 음성 파일 저장 폴더  
3. **audio_output**: STT 처리를 위한 중간 폴더
4. **script_output**: 변환된 텍스트 파일 저장 폴더

## 설치 방법

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 추가 설치 (Windows의 경우)
```bash
# FFmpeg 설치 (오디오 파일 처리용)
# https://ffmpeg.org/download.html 에서 다운로드 후 PATH 추가
```

## 사용 방법

### 1. 기본 사용법 (한국어 자동 감지)
```bash
python audio_pipeline.py
```

### 2. 일본어 영상/음성 처리
```bash
# 일본어 전용 스크립트
python audio_pipeline_japanese.py

# 또는 메인 스크립트에서 언어 지정
python -c "from audio_pipeline import main; main('ja')"
```

### 3. 다국어 지원 (대화형 선택)
```bash
python audio_pipeline_multilang.py
```

### 4. 지원 언어
- 🇰🇷 **한국어** (`ko`) - 기본값
- 🇯🇵 **일본어** (`ja`) - 일본 영상/음성
- 🇺🇸 **영어** (`en`) - English
- 🇨🇳 **중국어** (`zh`) - 中文
- 🇪🇸 **스페인어** (`es`) - Español
- 🇫🇷 **프랑스어** (`fr`) - Français
- 🇩🇪 **독일어** (`de`) - Deutsch
- 🇷🇺 **러시아어** (`ru`) - Русский

### 5. 파일 준비
- `audio_input` 폴더에 처리할 음성 파일들을 넣어주세요
- 지원 형식: WAV, MP3, M4A, FLAC, OGG

### 6. 실행 결과
- `audio_out`: 노이즈가 제거된 음성 파일
- `script_output`: 
  - `*_transcript.txt`: 타임스탬프가 포함된 전사 결과
  - `*_subtitle.srt`: SRT 자막 파일
  - `processing_summary_*.txt`: 처리 결과 요약
- `audio_pipeline.log`: 처리 과정 로그

## 주요 기능

### 노이즈 제거
- **모델**: SpeechBrain MetricGAN+
- **기능**: 배경 소음, 잡음 제거
- **출력**: 16kHz, 모노 채널 WAV 파일

### 음성 인식 (STT)
- **모델**: OpenAI Whisper
- **언어**: 8개 언어 지원 + 자동 감지
- **출력**: 
  - 타임스탬프가 포함된 텍스트 파일
  - SRT 자막 파일
  - 단어별 신뢰도 정보

## 시스템 요구사항

### 최소 요구사항
- Python 3.8+
- RAM: 4GB 이상
- 저장공간: 2GB 이상 (모델 다운로드용)

### 권장 요구사항
- GPU: CUDA 지원 (처리 속도 향상)
- RAM: 8GB 이상
- 저장공간: 5GB 이상

## 문제 해결

### 1. GPU 관련 오류
```bash
# CUDA 설치 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 모델 다운로드 실패
- 인터넷 연결 확인
- 방화벽 설정 확인
- 충분한 저장공간 확인

### 3. 오디오 파일 처리 오류
- FFmpeg 설치 확인
- 파일 형식 지원 여부 확인
- 파일 권한 확인

## 로그 확인

처리 과정의 상세한 로그는 `audio_pipeline.log` 파일에서 확인할 수 있습니다.

```bash
# 실시간 로그 확인
tail -f audio_pipeline.log
```

## 성능 최적화

### GPU 사용
- NVIDIA GPU + CUDA 설치 시 자동으로 GPU 사용
- 처리 속도 약 3-5배 향상

### 배치 처리
- 여러 파일을 한 번에 처리
- 모델 로딩 시간 최소화

## 사용 예시

### 일본어 영상 처리 예시
```bash
# 1. 일본어 영상에서 오디오 추출 (ffmpeg 사용)
ffmpeg -i japanese_video.mp4 -vn -acodec pcm_s16le -ar 16000 audio_input/japanese_audio.wav

# 2. 일본어 파이프라인 실행
python audio_pipeline_japanese.py

# 3. 결과 확인
# - audio_out/japanese_audio_denoised.wav (노이즈 제거된 음성)
# - script_output/japanese_audio_transcript.txt (타임스탬프 포함 전사)
# - script_output/japanese_audio_subtitle.srt (SRT 자막)
```

### 출력 파일 예시
**타임스탬프 포함 전사 파일 (`*_transcript.txt`)**:
```
📝 전체 텍스트
------------------------------
こんにちは、今日はいい天気ですね。

⏰ 타임스탬프별 텍스트
------------------------------
[00:00.000 → 00:02.500] こんにちは、
[00:02.500 → 00:05.000] 今日はいい天気ですね。

🔤 단어별 타임스탬프
------------------------------
[00:00.000-00:01.200] こんにちは (신뢰도: 0.95)
[00:01.200-00:01.500] 、 (신뢰도: 0.88)
[00:01.500-00:02.800] 今日は (신뢰도: 0.92)
```

**SRT 자막 파일 (`*_subtitle.srt`)**:
```
1
00:00:00,000 --> 00:02:500
こんにちは、

2
00:02:500 --> 00:05:000
今日はいい天気ですね。
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
