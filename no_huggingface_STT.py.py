#!/usr/bin/env python3
"""
오디오 파일 폴더 구조 즉시 생성
"""

import os
from pathlib import Path

# 현재 디렉토리 기준으로 폴더 생성
base_path = Path("d:/Translate-Project1")

# 생성할 폴더 목록
folders_to_create = [
    "audio_input",
    "audio_input/wav", 
    "audio_input/mp3",
    "audio_input/other",
    "audio_output",
    "audio_output/transcripts",
    "audio_output/cleaned", 
    "audio_output/processed",
    "temp",
    "samples"
]

print("📁 오디오 파일 관리용 폴더 생성 중...\n")

created_count = 0
for folder in folders_to_create:
    folder_path = base_path / folder
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {folder}")
        created_count += 1
        
        # 각 폴더에 README.md 생성
        readme_path = folder_path / "README.md"
        if not readme_path.exists():
            descriptions = {
                "audio_input": "입력 오디오 파일들을 저장하는 폴더",
                "audio_input/wav": "WAV 형식 오디오 파일 전용",
                "audio_input/mp3": "MP3 형식 오디오 파일 전용", 
                "audio_input/other": "기타 오디오 형식 (flac, m4a, aac, ogg)",
                "audio_output": "처리된 결과 파일들을 저장하는 폴더",
                "audio_output/transcripts": "STT 전사 결과 파일들 (txt, json, srt)",
                "audio_output/cleaned": "노이즈 제거된 오디오 파일들",
                "audio_output/processed": "기타 처리된 오디오 파일들",
                "temp": "임시 파일들을 저장하는 폴더",
                "samples": "샘플 및 테스트 오디오 파일들"
            }
            
            description = descriptions.get(folder, "오디오 관련 파일 저장")
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# {folder}\n\n{description}\n\n")
                f.write("## 사용 방법\n\n")
                if "input" in folder:
                    f.write("이 폴더에 처리할 오디오 파일들을 저장하세요.\n\n")
                    f.write("지원 형식: WAV, MP3, FLAC, M4A, AAC, OGG, MP4\n")
                elif "output" in folder:
                    f.write("처리 결과 파일들이 자동으로 저장됩니다.\n")
                elif folder == "temp":
                    f.write("임시 파일들이 저장됩니다. 주기적으로 정리하세요.\n")
                elif folder == "samples":
                    f.write("테스트용 샘플 파일들을 저장하세요.\n")
        
    except Exception as e:
        print(f"❌ {folder} 생성 실패: {e}")

print(f"\n🎉 총 {created_count}개 폴더가 생성되었습니다!")

# 사용 가이드 파일 생성
guide_content = """# 🎵 오디오 파일 폴더 사용 가이드

## 📁 폴더 구조
```
audio_input/          # 입력 오디오 파일들
├── wav/             # WAV 파일 (.wav)
├── mp3/             # MP3 파일 (.mp3)  
└── other/           # 기타 형식 (.flac, .m4a, .aac, .ogg)

audio_output/         # 처리 결과 파일들
├── transcripts/     # 전사 결과 (.txt, .json, .srt)
├── cleaned/         # 노이즈 제거된 오디오
└── processed/       # 기타 처리된 오디오

temp/                # 임시 파일들
samples/             # 샘플/테스트 파일들
```

## 🚀 사용 방법

### 1. 오디오 파일 저장
- WAV 파일 → `audio_input/wav/` 폴더
- MP3 파일 → `audio_input/mp3/` 폴더
- 기타 형식 → `audio_input/other/` 폴더

### 2. STT + 화자 분리 실행
```bash
# 특정 폴더 처리
python stt_diarization.py --input audio_input/wav/ --batch --output audio_output/transcripts/

# 모든 입력 파일 처리
python stt_diarization.py --input audio_input/ --batch --output audio_output/transcripts/
```

### 3. 결과 확인
- 전사 결과: `audio_output/transcripts/` 폴더
- 처리된 오디오: `audio_output/processed/` 폴더

## 💡 파일 명명 규칙
- 입력: `meeting_20241023.wav`, `interview_john.mp3`
- 출력: `meeting_20241023_transcript.txt`, `interview_john_transcript.json`

## 📋 주의사항
1. 원본 파일은 항상 백업해두세요
2. 대용량 파일은 처리 시간이 오래 걸립니다
3. `temp/` 폴더는 주기적으로 정리하세요
"""

guide_path = base_path / "AUDIO_FOLDERS_GUIDE.md"
with open(guide_path, 'w', encoding='utf-8') as f:
    f.write(guide_content)

print(f"📖 사용 가이드 생성: {guide_path}")

# 현재 생성된 폴더 목록 출력
print(f"\n📋 생성된 폴더 목록:")
for folder in folders_to_create:
    folder_path = base_path / folder
    if folder_path.exists():
        print(f"   ✅ {folder}")
    else:
        print(f"   ❌ {folder}")

print(f"\n💡 이제 다음과 같이 사용하세요:")
print(f"1. 오디오 파일을 해당 폴더에 저장")
print(f"2. python stt_diarization.py --input audio_input/ --batch 실행")
print(f"3. audio_output/transcripts/ 에서 결과 확인")

if __name__ == "__main__":
    pass
