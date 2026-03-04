"""
test_pipeline.py — інтеграційні тести для Pipeline.

Стратегія:
- Всі зовнішні залежності (AudioExtractor, Transcriber, тощо) замоковані
- Перевіряємо збереження/відновлення стану, логіку пропуску кроків,
  set_tts_choices, run_all
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import Pipeline, PipelineState, PipelineStep, PipelineError
from config import AppConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_SEGMENTS = [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0, "text": "Hello"},
    {"speaker": "SPEAKER_01", "start": 4.0, "end": 6.0, "text": "World"},
]
SAMPLE_TRANSLATED = [
    {**s, "translated_text": "Привет" if i == 0 else "Мир"}
    for i, s in enumerate(SAMPLE_SEGMENTS)
]
SAMPLE_META = {
    "SPEAKER_00": {"total_duration_sec": 120.0, "sample_count": 5, "sample_paths": ["/fake/s1.wav"], "tts_method": "gpt_sovits"},
    "SPEAKER_01": {"total_duration_sec": 15.0,  "sample_count": 2, "sample_paths": ["/fake/s2.wav"], "tts_method": "f5_tts"},
}
SAMPLE_TTS_RESULTS = {
    "SPEAKER_00": [{"segment_index": 0, "path": "/fake/tts/p0.wav", "duration_sec": 2.8}],
    "SPEAKER_01": [{"segment_index": 1, "path": "/fake/tts/p1.wav", "duration_sec": 2.1}],
}
SAMPLE_STRETCHED = {
    "SPEAKER_00": [{"segment_index": 0, "path": "/fake/stretched/p0.wav", "duration_sec": 3.0, "target_duration_sec": 3.0}],
    "SPEAKER_01": [{"segment_index": 1, "path": "/fake/stretched/p1.wav", "duration_sec": 2.0, "target_duration_sec": 2.0}],
}


def make_pipeline(tmp_path: Path, input_file: str = "/fake/video.mp4") -> Pipeline:
    return Pipeline(
        project_dir=tmp_path / "project",
        input_file=input_file,
        config=AppConfig(),
    )


from contextlib import contextmanager

@contextmanager
def mock_all_modules():
    """Контекст-менеджер для мокування всіх зовнішніх модулів пайплайну."""
    mock_extractor_instance = MagicMock()
    mock_extractor_instance.extract.return_value = Path("/fake/audio/clean.wav")
    mock_extractor = MagicMock(return_value=mock_extractor_instance)

    mock_transcriber_instance = MagicMock()
    mock_transcriber_instance.transcribe.return_value = SAMPLE_SEGMENTS
    mock_transcriber = MagicMock(return_value=mock_transcriber_instance)

    mock_collector_instance = MagicMock()
    mock_collector_instance.collect.return_value = SAMPLE_META
    mock_collector = MagicMock(return_value=mock_collector_instance)

    mock_translator_instance = MagicMock()
    mock_translator_instance.translate.return_value = SAMPLE_TRANSLATED
    mock_translator = MagicMock(return_value=mock_translator_instance)

    mock_tts_instance = MagicMock()
    mock_tts_instance.generate_all.return_value = SAMPLE_TTS_RESULTS
    mock_tts = MagicMock(return_value=mock_tts_instance)

    mock_stretcher_instance = MagicMock()
    mock_stretcher_instance.stretch_all.return_value = SAMPLE_STRETCHED
    mock_stretcher = MagicMock(return_value=mock_stretcher_instance)

    mock_mixer_instance = MagicMock()
    mock_mixer_instance.mix.return_value = Path("/fake/output/final.wav")
    mock_mixer = MagicMock(return_value=mock_mixer_instance)

    with (
        patch("pipeline.AudioExtractor", mock_extractor),
        patch("pipeline.Transcriber", mock_transcriber),
        patch("pipeline.ReferenceCollector", mock_collector),
        patch("pipeline.Translator", mock_translator),
        patch("pipeline.TTSEngine", mock_tts),
        patch("pipeline.TimeStretcher", mock_stretcher),
        patch("pipeline.Mixer", mock_mixer),
    ):
        yield


# ---------------------------------------------------------------------------
# PipelineState
# ---------------------------------------------------------------------------

class TestPipelineState:
    def test_to_dict_and_from_dict(self):
        state = PipelineState(
            input_file="/video.mp4",
            current_step=PipelineStep.TRANSCRIBE,
            segments=SAMPLE_SEGMENTS,
        )
        d = state.to_dict()
        restored = PipelineState.from_dict(d)
        assert restored.input_file == "/video.mp4"
        assert restored.current_step == PipelineStep.TRANSCRIBE
        assert restored.segments == SAMPLE_SEGMENTS

    def test_from_dict_ignores_unknown_keys(self):
        state = PipelineState.from_dict({"input_file": "/x.mp4", "unknown_key": "value"})
        assert state.input_file == "/x.mp4"


# ---------------------------------------------------------------------------
# Pipeline init / state persistence
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_creates_state_file_after_step(self, tmp_path: Path):
        with mock_all_modules():
            pipeline = make_pipeline(tmp_path)
            pipeline._step_extract_audio()

        state_path = tmp_path / "project" / "pipeline_state.json"
        assert state_path.exists()

    def test_loads_existing_state(self, tmp_path: Path):
        project_dir = tmp_path / "project"
        project_dir.mkdir(parents=True)
        state = PipelineState(
            input_file="/video.mp4",
            last_completed_step=PipelineStep.TRANSLATE,
            segments=SAMPLE_SEGMENTS,
            translated_segments=SAMPLE_TRANSLATED,
        )
        state_file = project_dir / "pipeline_state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f)

        pipeline = Pipeline(project_dir=project_dir)
        assert pipeline.state.last_completed_step == PipelineStep.TRANSLATE
        assert pipeline.state.segments == SAMPLE_SEGMENTS


# ---------------------------------------------------------------------------
# Step skipping (resume)
# ---------------------------------------------------------------------------

class TestStepSkipping:
    def test_skips_completed_steps(self, tmp_path: Path):
        with mock_all_modules():
            pipeline = make_pipeline(tmp_path)
            # Вручну позначаємо EXTRACT_AUDIO як виконаний
            pipeline.state.last_completed_step = PipelineStep.EXTRACT_AUDIO
            pipeline.state.audio_path = "/fake/audio.wav"

            # Транскрипція повинна бути викликана
            pipeline._step_transcribe = MagicMock()
            pipeline._run_steps([PipelineStep.EXTRACT_AUDIO, PipelineStep.TRANSCRIBE])

            # EXTRACT_AUDIO пропущено, TRANSCRIBE виконано
            pipeline._step_transcribe.assert_called_once()

    def test_is_step_done_false_for_empty_state(self, tmp_path: Path):
        pipeline = make_pipeline(tmp_path)
        assert not pipeline._is_step_done(PipelineStep.EXTRACT_AUDIO)

    def test_is_step_done_true_after_marking(self, tmp_path: Path):
        with mock_all_modules():
            pipeline = make_pipeline(tmp_path)
            pipeline.state.last_completed_step = PipelineStep.TRANSLATE
            assert pipeline._is_step_done(PipelineStep.EXTRACT_AUDIO)
            assert pipeline._is_step_done(PipelineStep.TRANSCRIBE)
            assert pipeline._is_step_done(PipelineStep.COLLECT_REFS)
            assert pipeline._is_step_done(PipelineStep.TRANSLATE)
            assert not pipeline._is_step_done(PipelineStep.GENERATE_TTS)


# ---------------------------------------------------------------------------
# set_tts_choices
# ---------------------------------------------------------------------------

class TestSetTtsChoices:
    def test_saves_choices_to_state(self, tmp_path: Path):
        with mock_all_modules():
            pipeline = make_pipeline(tmp_path)
            pipeline.set_tts_choices({"SPEAKER_00": "gpt_sovits", "SPEAKER_01": "f5_tts"})
            assert pipeline.state.tts_choices["SPEAKER_00"] == "gpt_sovits"

    def test_persists_choices_to_disk(self, tmp_path: Path):
        with mock_all_modules():
            pipeline = make_pipeline(tmp_path)
            pipeline.set_tts_choices({"SPEAKER_00": "f5_tts"})

        # Перезавантажуємо пайплайн і перевіряємо
        pipeline2 = Pipeline(project_dir=tmp_path / "project")
        assert pipeline2.state.tts_choices.get("SPEAKER_00") == "f5_tts"


# ---------------------------------------------------------------------------
# run_all (end-to-end з моками)
# ---------------------------------------------------------------------------

class TestRunAll:
    def test_run_all_returns_output_path(self, tmp_path: Path):
        with mock_all_modules():
            pipeline = make_pipeline(tmp_path, input_file="/fake/video.mp4")
            output = pipeline.run_all(tts_choices={"SPEAKER_00": "gpt_sovits", "SPEAKER_01": "f5_tts"})
            # Порівнюємо через Path щоб не залежати від роздільника (/ vs \)
            assert Path(output) == Path("/fake/output/final.wav")

    def test_run_all_auto_assigns_tts_if_none(self, tmp_path: Path):
        with mock_all_modules():
            pipeline = make_pipeline(tmp_path, input_file="/fake/video.mp4")
            pipeline.run_all()
            # Перевіряємо що tts_choices автоматично взяті з speakers_meta
            assert "SPEAKER_00" in pipeline.state.tts_choices
            assert pipeline.state.tts_choices["SPEAKER_00"] == "gpt_sovits"
            assert pipeline.state.tts_choices["SPEAKER_01"] == "f5_tts"

    def test_run_from_tts_raises_without_choices(self, tmp_path: Path):
        with mock_all_modules():
            pipeline = make_pipeline(tmp_path)
            with pytest.raises(PipelineError, match="tts_choices не задані"):
                pipeline.run_from_tts()

    def test_error_saves_state_with_error_step(self, tmp_path: Path):
        with mock_all_modules():
            pipeline = make_pipeline(tmp_path, input_file="/fake/video.mp4")
            pipeline._step_extract_audio = MagicMock(side_effect=RuntimeError("boom"))
            with pytest.raises(PipelineError):
                pipeline._run_steps([PipelineStep.EXTRACT_AUDIO])
            assert pipeline.state.current_step == PipelineStep.ERROR
            assert "boom" in pipeline.state.error_message


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self, tmp_path: Path):
        with mock_all_modules():
            pipeline = make_pipeline(tmp_path, input_file="/fake/video.mp4")
            pipeline.state.segments = SAMPLE_SEGMENTS
            pipeline.state.last_completed_step = PipelineStep.TRANSLATE
            pipeline.reset()
            assert pipeline.state.segments == []
            assert pipeline.state.last_completed_step == ""
            assert pipeline.state.input_file == "/fake/video.mp4"
