# transcribe_diarize.py
import os, argparse, pathlib
from pathlib import Path
import torch, whisperx
from whisperx.diarize import DiarizationPipeline
import huggingface_hub
import gc
import time
huggingface_hub.login("hf_")

# Windows 전용: PyTorch DLL 경로 우선
try:
    os.add_dll_directory(str(pathlib.Path(torch.__file__).parents[1] / "lib"))
except Exception:
    pass

# 옵션: 속도 향상(TF32). 필요 없으면 주석 처리.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def mmss(t):
    t = 0.0 if not isinstance(t, (int, float)) or t != t or t < 0 else float(t)
    m = int(t // 60)
    s = int(round(t - m * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m:02d}:{s:02d}"

def write_lines(p: Path, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def build_diar_pipeline(token: str, prefer_device: str):
    try:
        return DiarizationPipeline(use_auth_token=token, device=prefer_device)
    except OSError as e:
        print(f"[WARN] GPU diarization 실패: {e}\n[INFO] CPU로 폴백.")
        return DiarizationPipeline(use_auth_token=token, device="cpu")

def looks_blackwell(gpu_name: str):
    n = gpu_name.lower()
    keys = ["rtx 50", "rtx5", "5070", "5080", "5090", "blackwell", "gb2", "gb20"]
    return any(k in n for k in keys)

def cleanup_memory():
    """메모리 정리 함수"""
    print("🧹 메모리 정리 중...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(1)  # 시스템 안정화를 위한 대기

def cleanup_models(*models):
    """모델 객체들을 정리하는 함수"""
    for model in models:
        if model is not None:
            del model
    cleanup_memory()

def process_one(fp: Path, out_dir: Path, model, diar_pipe, device: str, batch: int,
                min_spk, max_spk, load_model_fn, compute_type: str):
    align_model = None
    meta = None
    
    try:
        print(f"🎵 처리 시작: {fp.name}")
        audio = whisperx.load_audio(str(fp))

        # 1) ASR(+언어감지). INT8 경로(cuBLAS) 실패 시 FP16으로 1회 자동 재시도.
        try:
            asr = model.transcribe(audio, batch_size=batch)
        except RuntimeError as e:
            msg = str(e).lower()
            if ("cublas_status_not_supported" in msg or "cublas" in msg) and "int8" in (compute_type or ""):
                print("[INFO] cuBLAS INT8 문제 감지. FP16으로 재시도.")
                model = load_model_fn("float16")
                asr = model.transcribe(audio, batch_size=batch)
            else:
                raise

        lang = asr.get("language", "ko")
        print(f"   - 감지된 언어: {lang}")

        # 2) 정렬
        align_model, meta = whisperx.load_align_model(language_code=lang, device=device)
        asr_aligned = whisperx.align(
            asr["segments"], align_model, meta, audio, device, return_char_alignments=False
        )

        # 3) 화자분리
        diar_kwargs = {}
        if min_spk is not None:
            diar_kwargs["min_speakers"] = min_spk
        if max_spk is not None:
            diar_kwargs["max_speakers"] = max_spk
        diar_segments = diar_pipe(audio, **diar_kwargs)
        asr_spk = whisperx.assign_word_speakers(diar_segments, asr_aligned)

        # 4) 저장: [mm:ss] 화자N : 텍스트
        spk_map, lines = {}, []
        for seg in sorted(asr_spk["segments"], key=lambda x: x.get("start", 0.0)):
            txt = (seg.get("text") or "").strip()
            if not txt:
                continue
            spk = seg.get("speaker", "UNK")
            if spk not in spk_map:
                # A, B, C, ... 형식으로 화자 이름 생성
                speaker_letter = chr(ord('A') + len(spk_map))
                spk_map[spk] = f"화자{speaker_letter}"
            lines.append(f"[{mmss(seg.get('start', 0.0))}] {spk_map[spk]} : {txt}")

        out_path = out_dir / f"{fp.stem}_transcript_{lang}.txt"
        write_lines(out_path, lines)
        
        print(f" 완료: {fp.name} -> {out_path.name}")
        return out_path
        
    except Exception as e:
        print(f"❌ 오류: {fp.name} - {e}")
        raise
    finally:
        # 파일별 처리 후 정리
        if align_model is not None:
            cleanup_models(align_model)
            align_model = None
        if meta is not None:
            del meta
            meta = None
        if 'audio' in locals():
            del audio
        cleanup_memory()

def main():
    ap = argparse.ArgumentParser(
        description="WhisperX + diarization 배치 처리 (다양한 오디오 형식 지원)",
        epilog="""
사용 예시:
  python whisperx_test.py --src audio_folder                    # 폴더의 모든 오디오 파일 처리
  python whisperx_test.py --src my_audio.wav                    # 단일 WAV 파일 처리
  python whisperx_test.py --src audio_folder --ext "*.mp3"      # MP3 파일만 처리
  python whisperx_test.py --src audio_folder --ext "*.wav"      # WAV 파일만 처리
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default="video_input", help="오디오/비디오 파일 또는 폴더 (mp4, wav, mp3, flac, m4a, aac, ogg 지원)")
    ap.add_argument("--out_dir", default="input", help="출력 폴더")
    ap.add_argument("--whisper_model", default="large-v3", help="Whisper 모델명")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--min_speakers", type=int, default=None)
    ap.add_argument("--max_speakers", type=int, default=None)
    ap.add_argument("--hf_token", default=None, help="Hugging Face read 토큰")
    ap.add_argument("--force_cpu_diar", action="store_true", help="화자분리만 CPU 강제")
    ap.add_argument("--compute_type", default=None, help='CUDA: float16|bfloat16|float32|int8_float16 등')
    ap.add_argument("--ext", default="*.mp4", help="검색 확장자 패턴 (예: *.mp4, *.wav, *.mp3)")
    ap.add_argument("--cleanup_interval", type=int, default=3, help="모델 정리 주기 (파일 개수)")
    args = ap.parse_args()

    token = (
        args.hf_token
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
    )

    # GPU cuDNN 문제 우회를 위해 CPU 사용
    device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else ""
    is_blackwell = device == "cuda" and looks_blackwell(gpu_name)

    # 기본 정밀도 선택: Blackwell(RTX 50xx)에서는 INT8 계열 회피
    if args.compute_type:
        compute_type = args.compute_type
    else:
        if device == "cuda":
            compute_type = "float16" if is_blackwell else "int8_float16"
        else:
            compute_type = "int8"

    def load_model_with(ct):
        return whisperx.load_model(args.whisper_model, device=device, compute_type=ct)

    # 모델 로드
    model = load_model_with(compute_type)

    diar_device = "cpu" if args.force_cpu_diar else device
    diar_pipe = build_diar_pipeline(token, diar_device)

    # 입력 수집
    src_path = Path(args.src)
    
    if src_path.is_file():
        files = [src_path]
    else:
        # 여러 오디오 형식 지원
        audio_extensions = ["*.mp4", "*.wav", "*.mp3", "*.flac", "*.m4a", "*.aac", "*.ogg"]
        files = []
        
        if args.ext != "*.mp4":  # 사용자가 특정 확장자를 지정한 경우
            files = sorted(src_path.glob(args.ext))
        else:  # 기본값인 경우 모든 오디오 형식 검색
            for ext in audio_extensions:
                files.extend(src_path.glob(ext))
            files = sorted(set(files))  # 중복 제거 및 정렬
    
    assert files, f"{src_path}에서 오디오 파일을 찾지 못함. 지원 형식: {', '.join(audio_extensions)}"
    
    # 찾은 파일들 표시
    print(f"\n📁 처리할 파일 {len(files)}개:")
    for i, fp in enumerate(files, 1):
        print(f"  {i}. {fp.name} ({fp.suffix.upper()})")
    print()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 배치 처리 시작: {len(files)}개 파일")
    successful = 0
    failed = 0

    for i, fp in enumerate(files, 1):
        try:
            print(f"\n{'='*50}")
            print(f"📊 진행률: {i}/{len(files)} ({i/len(files)*100:.1f}%)")
            
            outp = process_one(
                fp, out_dir, model, diar_pipe, device, args.batch_size,
                args.min_speakers, args.max_speakers, load_model_with, compute_type
            )
            successful += 1
            print(f"OK: {fp.name} -> {outp.name}")
            
        except Exception as e:
            failed += 1
            print(f"ERR: {fp.name}: {e}")
        
        # 주기적으로 전체 모델 정리 (메모리 누수 방지)
        if i % args.cleanup_interval == 0 and i < len(files):
            print(f"\n🔄 주기적 모델 정리 ({i}/{len(files)})")
            cleanup_models(model, diar_pipe)
            
            # 모델 재로드
            print("📥 모델 재로드 중...")
            model = load_model_with(compute_type)
            diar_pipe = build_diar_pipeline(token, diar_device)
            
            time.sleep(2)  # 시스템 안정화
    
    # 최종 결과 출력
    print(f"\n{'='*50}")
    print(f"🎉 배치 처리 완료!")
    print(f"   - 성공: {successful}개")
    print(f"   - 실패: {failed}개")
    print(f"   - 총 처리: {len(files)}개")
    
    # 최종 정리
    print("\n🧹 최종 정리...")
    cleanup_models(model, diar_pipe)

if __name__ == "__main__":
    main()



# # 기본 사용 (3파일마다 정리)
# python whisperx_test.py --src video_folder

# # 정리 주기 변경 (5파일마다)
# python whisperx_test.py --src video_folder --cleanup_interval 5

# # 단일 파일 처리
# python whisperx_test.py --src video.mp4