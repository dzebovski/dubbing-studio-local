🚀 Локальний AI-комбайн для автодубляжу відео (RTX 4080 16GB)
Цей документ описує створення локального Python-додатка з Terminal User Interface (TUI) для автоматичного дубляжу відео (з англійської на російську/українську) зі збереженням оригінальних голосів, інтонацій та таймінгів реплік.

🛠 1. Стек технологій та моделей
Враховуючи ліміт у 16 ГБ VRAM, головне правило системи: суворе почергове завантаження та вивантаження моделей з пам'яті GPU (torch.cuda.empty_cache()).

Інтерфейс (TUI): InquirerPy (для меню) + Rich (для логів та прогрес-барів).

Audio Extraction: ffmpeg-python.

STT & Diarization: WhisperX (модель Large-v3). Видає таймінги з точністю до мілісекунд і розпізнає різних спікеров (SPEAKER_00, SPEAKER_01).

LLM Translation: Ollama + Qwen-2.5-14B-Instruct (або Llama-3.1-8B-Instruct). Переклад субтитрів із збереженням кількості складів.

TTS (Голос):

Zero-Shot (Швидко/Мало даних): F5-TTS.

Fine-Tuning (Якісно/Багато даних): GPT-SoVITS (виклик CLI для тренування).

Time-Stretching: pyrubberband (або ffmpeg atempo), щоб підігнати довжину згенерованого звуку рівно під довжину оригінального шматка без зміни тональності голосу (Pitch Shift).

📋 2. Промпт (ТЗ) для AI-кодера
Скопіюй цей текст і відправ у Claude 3.5 Sonnet або GPT-4o, коли будеш генерувати модулі системи:

Роль: Ти Senior AI/Python розробник.
Завдання: Напиши локальний Python-додаток для автодубляжу з TUI-інтерфейсом (використовуй InquirerPy та Rich). Система працює на RTX 4080 (16GB VRAM), тому моделі повинні завантажуватись та вивантажуватись послідовно.

Пайплайн системи:

Вхід: відео або аудіо файл. Скрипт витягує чисте аудіо через ffmpeg.

Транскрипція: Запуск whisperX для транскрипції та діаризації. На виході — розбивка по репліках (start, end, text, speaker_id). Після цього обов'язково torch.cuda.empty_cache().

Збір референсів: Скрипт автоматично нарізає найкращі аудіо-семпли для кожного знайденого speaker_id.

Переклад: Парсинг реплік та їх відправка через API до локального Ollama (переклад з англійської на російську з промптом на збереження довжини фрази, тобто "isochrony").

Вибір TTS (через TUI): Для кожного спікера запитати користувача: використовувати F5-TTS (Zero-shot для 10-15 сек референсу) чи GPT-SoVITS (автоматичний виклик скриптів тренування, якщо референс > 1 хв).

Генерація: Озвучка перекладених фраз вибраним TTS.

Синхронізація (Time-Stretching): Порівняти тривалість згенерованої фрази з оригіналом (з WhisperX). Використати pyrubberband або ffmpeg для розтягнення/стиснення згенерованого аудіо, щоб воно мілісекунда-в-мілісекунду влізло у відведений час.

Зведення: Злиття всіх підігнаних аудіо-шматків в одну фінальну доріжку (pydub) і мікс з оригінальним відео (ffmpeg).

Напиши модульну архітектуру (ООП), де кожен етап — це окремий клас. Почни з базового каркасу (main.py) з реалізацією TUI та логікою виклику методів.

💻 3. Базовий каркас коду (Скелет Пайплайну)
Це структура main.py, яка показує логіку роботи комбайну.

code
Python
import os
import torch
from InquirerPy import inquirer
from rich.console import Console
from rich.progress import track

console = Console()

class AudioPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.project_dir = "output_project"
        os.makedirs(self.project_dir, exist_ok=True)
        self.subs_data =[]

    def clear_vram(self):
        """Очищення пам'яті відеокарти між завантаженнями різних моделей."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            console.print("[green]VRAM очищено.[/green]")

    def run_whisperx(self):
        console.print("[bold blue]1. Запуск WhisperX (Транскрипція та Діаризація)...[/bold blue]")
        # Тут логіка виклику whisperX
        # Формат виходу: self.subs_data =[{'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 3.5, 'text': 'Hello world'}]
        self.clear_vram()

    def translate_with_ollama(self):
        console.print("[bold blue]2. Переклад через Ollama (Збереження таймінгів)...[/bold blue]")
        # Тут API-виклик до Ollama для кожного елемента self.subs_data
        # Додає ключ 'translated_text'

    def choose_tts_and_generate(self):
        console.print("[bold blue]3. Вибір методу TTS та генерація...[/bold blue]")
        # Знаходимо всіх унікальних спікерів
        speakers =["SPEAKER_00", "SPEAKER_01"] # Мок-дані (mock)
        
        for speaker in speakers:
            choice = inquirer.select(
                message=f"Виберіть метод TTS для {speaker}:",
                choices=[
                    "F5-TTS (Zero-shot, швидко, мало матеріалу)",
                    "GPT-SoVITS (Fine-tuning, якісно, багато матеріалу)"
                ],
            ).execute()

            if "F5-TTS" in choice:
                self.run_f5_tts(speaker)
            else:
                self.run_gpt_sovits_training(speaker)
        
        self.clear_vram()

    def run_f5_tts(self, speaker):
        console.print(f"Генерація {speaker} через F5-TTS...")
        # Логіка F5-TTS

    def run_gpt_sovits_training(self, speaker):
        console.print(f"Тренування та генерація {speaker} через GPT-SoVITS...")
        # Виклик subprocess для тренування моделі SoVITS
    
    def apply_time_stretching(self):
        console.print("[bold blue]4. Time-Stretching (підгонка таймінгів під оригінал)...[/bold blue]")
        # Порівняння тривалості оригінал vs генерація
        # Використання pyrubberband для прискорення/сповільнення без зміни пітчу

    def merge_audio(self):
        console.print("[bold blue]5. Зведення фінальної доріжки...[/bold blue]")
        # Використання pydub для розміщення шматків на таймлайні

def main():
    console.print("[bold magenta]=== AI Auto-Dubbing Pipeline (RTX 4080) ===[/bold magenta]")
    
    video_file = inquirer.filepath(
        message="Вкажіть шлях до відео/аудіо файлу:",
        default="./",
    ).execute()

    pipeline = AudioPipeline(video_file)
    
    pipeline.run_whisperx()
    pipeline.translate_with_ollama()
    pipeline.choose_tts_and_generate()
    pipeline.apply_time_stretching()
    pipeline.merge_audio()

    console.print("[bold green]✅ Дубляж успішно завершено! Файл збережено.[/bold green]")

if __name__ == "__main__":
    main()