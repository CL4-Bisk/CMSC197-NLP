"""Unit tests for Pipeline 1, Step 7 (src/converter/segmenter.py)."""
import pytest

from src.converter.segmenter import (
    SegmenterConfig,
    SegmentationSignal,
    _detect_time_phrases,
    _detect_setting,
    _character_jaccard,
    _time_continuous,
    _score_boundary,
    _find_boundaries,
    segment_events,
    DEFAULT_SETTING_PROTOTYPES,
    TimeOfDay,
)
from src.converter.preprocess import RawLine
from src.converter.classifier import LineType
from src.converter.characters import CharacterEnrichedLine
from src.schema import CharacterEntry


def make_enriched_line(text: str, chapter: int = 1, paragraph: int = 1,
                       line_type: LineType = LineType.NARRATION,
                       speaker_id: str = None, **kwargs) -> CharacterEnrichedLine:
    raw = RawLine(
        chapter=chapter,
        paragraph=paragraph,
        text=text,
        quoted=False,
        italic=False,
        char_start=0,
        char_end=len(text),
    )
    return CharacterEnrichedLine(raw=raw, line_type=line_type, speaker_id=speaker_id, **kwargs)


class TestTimeDetection:
    def test_morning_phrases(self):
        assert _detect_time_phrases("At dawn he woke") == TimeOfDay.MORNING
        assert _detect_time_phrases("Early morning light") == TimeOfDay.MORNING
        assert _detect_time_phrases("Sunrise painted the sky") == TimeOfDay.MORNING

    def test_day_phrases(self):
        assert _detect_time_phrases("At midday the sun was high") == TimeOfDay.DAY
        assert _detect_time_phrases("Noon approached") == TimeOfDay.DAY

    def test_night_phrases(self):
        assert _detect_time_phrases("Midnight struck") == TimeOfDay.NIGHT
        assert _detect_time_phrases("The darkness was complete") == TimeOfDay.NIGHT

    def test_no_phrases(self):
        assert _detect_time_phrases("He walked slowly") is None


class TestSettingDetection:
    def test_setting_prototypes_exist(self):
        assert len(DEFAULT_SETTING_PROTOTYPES) > 20
        assert "forest at night" in DEFAULT_SETTING_PROTOTYPES
        assert "sect courtyard" in DEFAULT_SETTING_PROTOTYPES

    @pytest.mark.slow
    def test_detect_setting(self):
        # This requires sentence-transformers model
        setting = _detect_setting("The forest was dark and quiet at night")
        assert setting in DEFAULT_SETTING_PROTOTYPES


class TestCharacterJaccard:
    def test_same_speaker(self):
        line1 = make_enriched_line("Hello", speaker_id="char_001")
        line2 = make_enriched_line("World", speaker_id="char_001")
        assert _character_jaccard(line1, line2) == 1.0

    def test_different_speaker(self):
        line1 = make_enriched_line("Hello", speaker_id="char_001")
        line2 = make_enriched_line("World", speaker_id="char_002")
        assert _character_jaccard(line1, line2) == 0.0

    def test_no_speaker(self):
        line1 = make_enriched_line("Narration")
        line2 = make_enriched_line("More narration")
        assert _character_jaccard(line1, line2) == 1.0


class TestTimeContinuous:
    def test_same_time(self):
        assert _time_continuous(TimeOfDay.MORNING, TimeOfDay.MORNING) is True

    def test_adjacent_times(self):
        assert _time_continuous(TimeOfDay.MORNING, TimeOfDay.DAY) is True
        assert _time_continuous(TimeOfDay.SUNSET, TimeOfDay.NIGHT) is True

    def test_non_adjacent(self):
        assert _time_continuous(TimeOfDay.MORNING, TimeOfDay.NIGHT) is False

    def test_none_handling(self):
        assert _time_continuous(None, TimeOfDay.MORNING) is True
        assert _time_continuous(TimeOfDay.MORNING, None) is True


class TestBoundaryScoring:
    def test_scene_break_hard(self):
        signal = SegmentationSignal(index=0, scene_break=True)
        score = _score_boundary(signal, SegmenterConfig())
        assert score == 1.0

    def test_embedding_drop(self):
        signal = SegmentationSignal(index=0, embedding_drop=0.8)
        score = _score_boundary(signal, SegmenterConfig())
        assert score >= 0.24  # 0.8 * 0.3

    def test_weak_label_below_threshold(self):
        signal = SegmentationSignal(index=0, embedding_drop=0.1)
        score = _score_boundary(signal, SegmenterConfig(threshold=0.5))
        assert score < 0.5


class TestBoundaryFinding:
    def test_min_segment_length(self):
        signals = [
            SegmentationSignal(index=0, embedding_drop=0.9),
            SegmentationSignal(index=1, embedding_drop=0.9),
            SegmentationSignal(index=2),
        ]
        boundaries = _find_boundaries(signals, SegmenterConfig(threshold=0.5, min_segment_length=2))
        # Should enforce min_segment_length
        assert len(boundaries) >= 2

    def test_scene_break_boundary(self):
        signals = [
            SegmentationSignal(index=0, embedding_drop=0.1),
            SegmentationSignal(index=1, scene_break=True),
            SegmentationSignal(index=2),
        ]
        boundaries = _find_boundaries(signals, SegmenterConfig(threshold=0.5, use_weak_labels=True))
        assert 2 in boundaries  # After scene break


class TestSegmenterConfig:
    def test_defaults(self):
        cfg = SegmenterConfig()
        assert cfg.method == "embedding"
        assert cfg.threshold == 0.5
        assert cfg.use_weak_labels is True
        assert cfg.pov_tracking is True

    def test_custom_values(self):
        cfg = SegmenterConfig(
            method="texttiling",
            threshold=0.7,
            window_size=5,
            pov_tracking=False,
        )
        assert cfg.method == "texttiling"
        assert cfg.threshold == 0.7
        assert cfg.window_size == 5
        assert cfg.pov_tracking is False


class TestIntegration:
    @pytest.mark.slow
    def test_segment_events_simple(self):
        """Test full segmentation pipeline."""
        from src.converter.preprocess import preprocess
        from src.converter.classifier import classify_lines
        from src.converter.characters import extract_characters, CharacterConfig
        from src.converter.emotions import tag_emotions, EmotionConfig

        text = (
            "Chapter 1\n\n"
            "Lin Feng stood in the courtyard. \"I will win,\" he said.\n\n"
            "***\n\n"
            "The forest was dark. A shadow approached.\n\n"
            "\"Who goes there?\" Lin Feng asked."
        )
        chapters, lines = preprocess(text, is_text=True)
        line_types = classify_lines(lines)
        char_cfg = CharacterConfig(ner_model="spacy_sm")
        characters, enriched = extract_characters(chapters, lines, line_types, char_cfg)
        emo_cfg = EmotionConfig(model="nrc")
        emotions = tag_emotions(lines, line_types, emo_cfg)
        for e, emos in zip(enriched, emotions):
            e.emotions = emos

        seg_cfg = SegmenterConfig()
        events = segment_events(enriched, characters, seg_cfg)

        # Should have at least 2 events (separated by ***)
        assert len(events) >= 2
        for evt in events:
            assert isinstance(evt, type(events[0]))  # PlotEvent
            assert len(evt.scenarios) >= 1

    @pytest.mark.slow
    def test_pov_tracking(self):
        """Test POV-based scenario splitting.

        Note: POV tracking requires monologues to be separate paragraphs
        (preprocess only detects whole-line italics).
        """
        from src.converter.preprocess import preprocess
        from src.converter.classifier import classify_lines
        from src.converter.characters import extract_characters, CharacterConfig
        from src.converter.emotions import tag_emotions, EmotionConfig

        text = (
            "Chapter 1\n\n"
            "Lin Feng drew his sword.\n\n"
            "*I must win.*\n\n"
            "He struck.\n\n"
            "*The elder watched calmly.*\n\n"
            "He nodded.\n\n"
            "Lin Feng smiled.\n\n"
            "*Victory is mine.*"
        )
        chapters, lines = preprocess(text, is_text=True)
        line_types = classify_lines(lines)
        char_cfg = CharacterConfig(ner_model="spacy_sm")
        characters, enriched = extract_characters(chapters, lines, line_types, char_cfg)
        emo_cfg = EmotionConfig(model="nrc")
        emotions = tag_emotions(lines, line_types, emo_cfg)
        for e, emos in zip(enriched, emotions):
            e.emotions = emos

        seg_cfg = SegmenterConfig(pov_tracking=True, min_pov_run_lines=2)
        events = segment_events(enriched, characters, seg_cfg)

        # Should detect POV shifts (Lin Feng -> Elder -> Lin Feng)
        total_scenarios = sum(len(e.scenarios) for e in events)
        assert total_scenarios >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])