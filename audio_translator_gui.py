#!/usr/bin/env python3
"""
Audio Translator GUI Application
LiveCaptions-Translator 스타일의 음성 번역 응용프로그램
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import webbrowser

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

try:
    from audio_pipeline import AudioPipeline
    from googletrans import Translator
except ImportError as e:
    print(f"Import 오류: {e}")
    print("필요한 패키지를 설치해주세요: pip install googletrans==4.0.0rc1")

class AudioTranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Translator - LiveCaptions Style")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # 스타일 설정
        self.setup_styles()
        
        # 변수 초기화
        self.pipeline = None
        self.translator = None
        self.current_file = None
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
        # GUI 구성요소 생성
        self.create_widgets()
        
        # 번역기 초기화
        self.init_translator()
        
        # 큐 처리 시작
        self.process_queue()
    
    def setup_styles(self):
        """GUI 스타일 설정"""
        style = ttk.Style()
        
        # 다크 테마 색상
        self.colors = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'accent': '#0078d4',
            'success': '#107c10',
            'warning': '#ff8c00',
            'error': '#d13438',
            'card': '#3c3c3c',
            'border': '#5a5a5a'
        }
        
        # 루트 배경색
        self.root.configure(bg=self.colors['bg'])
        
        # 스타일 구성
        style.theme_use('clam')
        style.configure('Card.TFrame', background=self.colors['card'], relief='raised', borderwidth=1)
        style.configure('Title.TLabel', background=self.colors['bg'], foreground=self.colors['fg'], 
                       font=('Segoe UI', 16, 'bold'))
        style.configure('Subtitle.TLabel', background=self.colors['bg'], foreground=self.colors['fg'], 
                       font=('Segoe UI', 10))
        style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'))
    
    def create_widgets(self):
        """GUI 위젯 생성"""
        # 메인 컨테이너
        main_frame = ttk.Frame(self.root, style='Card.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 헤더
        self.create_header(main_frame)
        
        # 파일 선택 영역
        self.create_file_section(main_frame)
        
        # 설정 영역
        self.create_settings_section(main_frame)
        
        # 진행 상황 영역
        self.create_progress_section(main_frame)
        
        # 결과 영역
        self.create_results_section(main_frame)
        
        # 하단 버튼
        self.create_bottom_buttons(main_frame)
    
    def create_header(self, parent):
        """헤더 영역 생성"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', pady=(0, 20))
        
        # 제목
        title_label = ttk.Label(header_frame, text="🎵 Audio Translator", style='Title.TLabel')
        title_label.pack(side='left')
        
        # 부제목
        subtitle_label = ttk.Label(header_frame, 
                                  text="음성 파일을 텍스트로 변환하고 번역합니다", 
                                  style='Subtitle.TLabel')
        subtitle_label.pack(side='left', padx=(10, 0))
        
        # GitHub 링크
        github_btn = ttk.Button(header_frame, text="GitHub", 
                               command=lambda: webbrowser.open("https://github.com/yhs5476/capstone_1"))
        github_btn.pack(side='right')
    
    def create_file_section(self, parent):
        """파일 선택 영역 생성"""
        file_frame = ttk.LabelFrame(parent, text="📁 파일 선택", padding=10)
        file_frame.pack(fill='x', pady=(0, 10))
        
        # 파일 경로 표시
        self.file_path_var = tk.StringVar(value="파일을 선택해주세요...")
        file_path_label = ttk.Label(file_frame, textvariable=self.file_path_var, 
                                   background=self.colors['card'], foreground=self.colors['fg'])
        file_path_label.pack(fill='x', pady=(0, 10))
        
        # 버튼 프레임
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill='x')
        
        # 파일 선택 버튼
        select_btn = ttk.Button(btn_frame, text="📂 파일 선택", 
                               command=self.select_file, style='Accent.TButton')
        select_btn.pack(side='left', padx=(0, 10))
        
        # 지원 형식 안내
        format_label = ttk.Label(btn_frame, 
                                text="지원 형식: MP3, WAV, MP4, AVI, MOV", 
                                style='Subtitle.TLabel')
        format_label.pack(side='left')
    
    def create_settings_section(self, parent):
        """설정 영역 생성"""
        settings_frame = ttk.LabelFrame(parent, text="⚙️ 설정", padding=10)
        settings_frame.pack(fill='x', pady=(0, 10))
        
        # 설정 그리드
        settings_grid = ttk.Frame(settings_frame)
        settings_grid.pack(fill='x')
        
        # 소스 언어
        ttk.Label(settings_grid, text="소스 언어:", style='Subtitle.TLabel').grid(
            row=0, column=0, sticky='w', padx=(0, 10))
        
        self.source_lang_var = tk.StringVar(value="자동 감지")
        source_combo = ttk.Combobox(settings_grid, textvariable=self.source_lang_var, 
                                   values=["자동 감지", "한국어", "일본어", "영어", "중국어", "스페인어"], 
                                   state="readonly", width=15)
        source_combo.grid(row=0, column=1, sticky='w', padx=(0, 20))
        
        # 타겟 언어
        ttk.Label(settings_grid, text="번역 언어:", style='Subtitle.TLabel').grid(
            row=0, column=2, sticky='w', padx=(0, 10))
        
        self.target_lang_var = tk.StringVar(value="한국어")
        target_combo = ttk.Combobox(settings_grid, textvariable=self.target_lang_var,
                                   values=["한국어", "일본어", "영어", "중국어", "스페인어", "프랑스어", "독일어"],
                                   state="readonly", width=15)
        target_combo.grid(row=0, column=3, sticky='w', padx=(0, 20))
        
        # 화자분리 옵션
        self.speaker_separation_var = tk.BooleanVar(value=True)
        speaker_check = ttk.Checkbutton(settings_grid, text="화자 분리", 
                                       variable=self.speaker_separation_var)
        speaker_check.grid(row=0, column=4, sticky='w')
        
        # 노이즈 제거 옵션
        self.denoise_var = tk.BooleanVar(value=True)
        denoise_check = ttk.Checkbutton(settings_grid, text="노이즈 제거", 
                                       variable=self.denoise_var)
        denoise_check.grid(row=0, column=5, sticky='w', padx=(20, 0))
    
    def create_progress_section(self, parent):
        """진행 상황 영역 생성"""
        progress_frame = ttk.LabelFrame(parent, text="📊 진행 상황", padding=10)
        progress_frame.pack(fill='x', pady=(0, 10))
        
        # 진행률 바
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.pack(fill='x', pady=(0, 10))
        
        # 상태 텍스트
        self.status_var = tk.StringVar(value="대기 중...")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, 
                                style='Subtitle.TLabel')
        status_label.pack()
    
    def create_results_section(self, parent):
        """결과 영역 생성"""
        results_frame = ttk.LabelFrame(parent, text="📝 결과", padding=10)
        results_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # 탭 위젯
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # 원본 텍스트 탭
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="원본 텍스트")
        
        self.original_text = scrolledtext.ScrolledText(
            self.original_frame, wrap=tk.WORD, height=10,
            bg=self.colors['card'], fg=self.colors['fg'], 
            font=('Consolas', 10))
        self.original_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 번역 텍스트 탭
        self.translated_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.translated_frame, text="번역 텍스트")
        
        self.translated_text = scrolledtext.ScrolledText(
            self.translated_frame, wrap=tk.WORD, height=10,
            bg=self.colors['card'], fg=self.colors['fg'], 
            font=('Consolas', 10))
        self.translated_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 로그 탭
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="로그")
        
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, wrap=tk.WORD, height=10,
            bg=self.colors['card'], fg=self.colors['fg'], 
            font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_bottom_buttons(self, parent):
        """하단 버튼 영역 생성"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', pady=(10, 0))
        
        # 처리 시작 버튼
        self.process_btn = ttk.Button(button_frame, text="🚀 처리 시작", 
                                     command=self.start_processing, 
                                     style='Accent.TButton')
        self.process_btn.pack(side='left', padx=(0, 10))
        
        # 중지 버튼
        self.stop_btn = ttk.Button(button_frame, text="⏹️ 중지", 
                                  command=self.stop_processing, state='disabled')
        self.stop_btn.pack(side='left', padx=(0, 10))
        
        # 결과 저장 버튼
        self.save_btn = ttk.Button(button_frame, text="💾 결과 저장", 
                                  command=self.save_results, state='disabled')
        self.save_btn.pack(side='left', padx=(0, 10))
        
        # 설정 저장/불러오기
        ttk.Button(button_frame, text="⚙️ 설정 저장", 
                  command=self.save_settings).pack(side='right', padx=(10, 0))
        ttk.Button(button_frame, text="📁 설정 불러오기", 
                  command=self.load_settings).pack(side='right')
    
    def init_translator(self):
        """번역기 초기화"""
        try:
            self.translator = Translator()
            self.log_message("번역기 초기화 완료")
        except Exception as e:
            self.log_message(f"번역기 초기화 실패: {e}", "ERROR")
    
    def select_file(self):
        """파일 선택"""
        filetypes = [
            ("오디오 파일", "*.mp3 *.wav *.m4a *.flac"),
            ("비디오 파일", "*.mp4 *.avi *.mov *.mkv"),
            ("모든 파일", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="음성/비디오 파일 선택",
            filetypes=filetypes
        )
        
        if filename:
            self.current_file = filename
            self.file_path_var.set(filename)
            self.log_message(f"파일 선택됨: {Path(filename).name}")
    
    def start_processing(self):
        """처리 시작"""
        if not self.current_file:
            messagebox.showwarning("경고", "먼저 파일을 선택해주세요.")
            return
        
        if not os.path.exists(self.current_file):
            messagebox.showerror("오류", "선택한 파일이 존재하지 않습니다.")
            return
        
        # UI 상태 변경
        self.is_processing = True
        self.process_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.save_btn.config(state='disabled')
        
        # 결과 텍스트 초기화
        self.original_text.delete(1.0, tk.END)
        self.translated_text.delete(1.0, tk.END)
        
        # 백그라운드 처리 시작
        thread = threading.Thread(target=self.process_audio, daemon=True)
        thread.start()
    
    def stop_processing(self):
        """처리 중지"""
        self.is_processing = False
        self.log_message("처리 중지됨")
        self.reset_ui_state()
    
    def process_audio(self):
        """오디오 처리 (백그라운드)"""
        try:
            # 파이프라인 초기화
            self.processing_queue.put(("status", "파이프라인 초기화 중..."))
            self.processing_queue.put(("progress", 10))
            
            if not self.pipeline:
                self.pipeline = AudioPipeline(use_gpu=True)
            
            # 파일 처리
            self.processing_queue.put(("status", "음성 인식 중..."))
            self.processing_queue.put(("progress", 30))
            
            # 음성 인식 실행
            result_text = self.pipeline.process_single_file(self.current_file)
            
            if not self.is_processing:
                return
            
            self.processing_queue.put(("status", "번역 중..."))
            self.processing_queue.put(("progress", 70))
            
            # 번역 실행
            translated_text = self.translate_text(result_text)
            
            if not self.is_processing:
                return
            
            # 결과 표시
            self.processing_queue.put(("original", result_text))
            self.processing_queue.put(("translated", translated_text))
            self.processing_queue.put(("status", "처리 완료!"))
            self.processing_queue.put(("progress", 100))
            self.processing_queue.put(("complete", True))
            
        except Exception as e:
            self.processing_queue.put(("error", str(e)))
    
    def translate_text(self, text):
        """텍스트 번역"""
        if not self.translator or not text.strip():
            return text
        
        try:
            # 언어 코드 매핑
            lang_map = {
                "한국어": "ko", "일본어": "ja", "영어": "en", 
                "중국어": "zh", "스페인어": "es", "프랑스어": "fr", "독일어": "de"
            }
            
            target_lang = lang_map.get(self.target_lang_var.get(), "ko")
            
            # 소스 언어가 타겟 언어와 같으면 번역하지 않음
            if self.source_lang_var.get() != "자동 감지":
                source_lang = lang_map.get(self.source_lang_var.get(), "auto")
                if source_lang == target_lang:
                    return text
            
            # 번역 실행
            result = self.translator.translate(text, dest=target_lang)
            return result.text
            
        except Exception as e:
            self.log_message(f"번역 오류: {e}", "ERROR")
            return text
    
    def process_queue(self):
        """큐 처리"""
        try:
            while True:
                msg_type, data = self.processing_queue.get_nowait()
                
                if msg_type == "status":
                    self.status_var.set(data)
                elif msg_type == "progress":
                    self.progress_var.set(data)
                elif msg_type == "original":
                    self.original_text.insert(tk.END, data)
                elif msg_type == "translated":
                    self.translated_text.insert(tk.END, data)
                elif msg_type == "log":
                    self.log_message(data)
                elif msg_type == "error":
                    self.log_message(f"오류: {data}", "ERROR")
                    messagebox.showerror("처리 오류", f"처리 중 오류가 발생했습니다:\n{data}")
                    self.reset_ui_state()
                elif msg_type == "complete":
                    self.reset_ui_state()
                    self.save_btn.config(state='normal')
                    messagebox.showinfo("완료", "처리가 완료되었습니다!")
                    
        except queue.Empty:
            pass
        
        # 100ms마다 큐 확인
        self.root.after(100, self.process_queue)
    
    def reset_ui_state(self):
        """UI 상태 초기화"""
        self.process_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress_var.set(0)
        self.status_var.set("대기 중...")
    
    def save_results(self):
        """결과 저장"""
        if not self.current_file:
            return
        
        try:
            base_path = Path(self.current_file).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 원본 텍스트 저장
            original_file = f"script_output/{base_path}_original_{timestamp}.txt"
            with open(original_file, 'w', encoding='utf-8') as f:
                f.write(self.original_text.get(1.0, tk.END))
            
            # 번역 텍스트 저장
            translated_file = f"script_output/{base_path}_translated_{timestamp}.txt"
            with open(translated_file, 'w', encoding='utf-8') as f:
                f.write(self.translated_text.get(1.0, tk.END))
            
            self.log_message(f"결과 저장 완료: {original_file}, {translated_file}")
            messagebox.showinfo("저장 완료", f"결과가 저장되었습니다:\n{original_file}\n{translated_file}")
            
        except Exception as e:
            self.log_message(f"저장 오류: {e}", "ERROR")
            messagebox.showerror("저장 오류", f"결과 저장 중 오류가 발생했습니다:\n{e}")
    
    def save_settings(self):
        """설정 저장"""
        settings = {
            "source_language": self.source_lang_var.get(),
            "target_language": self.target_lang_var.get(),
            "speaker_separation": self.speaker_separation_var.get(),
            "denoise": self.denoise_var.get()
        }
        
        try:
            with open("settings.json", 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            self.log_message("설정 저장 완료")
            messagebox.showinfo("설정 저장", "설정이 저장되었습니다.")
        except Exception as e:
            self.log_message(f"설정 저장 오류: {e}", "ERROR")
    
    def load_settings(self):
        """설정 불러오기"""
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                self.source_lang_var.set(settings.get("source_language", "자동 감지"))
                self.target_lang_var.set(settings.get("target_language", "한국어"))
                self.speaker_separation_var.set(settings.get("speaker_separation", True))
                self.denoise_var.set(settings.get("denoise", True))
                
                self.log_message("설정 불러오기 완료")
                messagebox.showinfo("설정 불러오기", "설정이 불러와졌습니다.")
            else:
                messagebox.showwarning("설정 파일 없음", "저장된 설정 파일이 없습니다.")
        except Exception as e:
            self.log_message(f"설정 불러오기 오류: {e}", "ERROR")
    
    def log_message(self, message, level="INFO"):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # 콘솔에도 출력
        print(log_entry.strip())

def main():
    """메인 함수"""
    root = tk.Tk()
    app = AudioTranslatorGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("애플리케이션 종료")

if __name__ == "__main__":
    main()
