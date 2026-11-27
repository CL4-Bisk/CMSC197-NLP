"""Unit tests for Pipeline 1, Step 6 (src/converter/emotions.py)."""
import pytest

from src.converter.emotions import (
    EmotionConfig,
    tag_emotions,
    tag_emotions_simple,
    map_to_renpy_expression,
    _get_nrc_lexicon,
    _merge_triggers,
    EMOTION_PRIORITY,
)
from src.converter.preprocess import RawLine, preprocess
from src.converter.classifier import classify_lines, LineType
from src.schema import EmotionTag, EmotionTrigger


def make_raw_line(text: str, **kwargs) -> RawLine:
    defaults = dict(chapter=1, paragraph=1, quoted=False, italic=False, char_start=0, char_end=len(text))
    defaults.update(kwargs)
    return RawLine(text=text, **defaults)


class TestNRCMapping:
    def test_canonical_mapping_loaded(self):
        from src.converter.emotions import _load_canonical_mapping
        mapping = _load_canonical_mapping()
        assert "joy" in mapping
        assert mapping["joy"] == "joy"
        assert mapping["anger"] == "anger"
        assert mapping["fear"] == "fear"

    def test_priority_order(self):
        # Fear should have highest priority for tiebreaking
        assert EMOTION_PRIORITY["fear"] > EMOTION_PRIORITY["anger"]
        assert EMOTION_PRIORITY["anger"] > EMOTION_PRIORITY["joy"]


class TestNRCLexicon:
    def test_lexicon_loads(self):
        lexicon = _get_nrc_lexicon()
        assert isinstance(lexicon, dict)
        assert len(lexicon) > 5000
        assert "fear" in lexicon
        assert "anger" in lexicon["fear"]
        assert "joy" in lexicon["happy"]


class TestMergeTriggers:
    def test_merge_adjacent(self):
        triggers = [
            EmotionTrigger(text="happy", start=0, end=5),
            EmotionTrigger(text="joyful", start=6, end=12),
        ]
        merged = _merge_triggers(triggers)
        assert len(merged) == 1
        assert merged[0].text == "happy joyful"
        assert merged[0].start == 0
        assert merged[0].end == 12

    def test_non_adjacent_not_merged(self):
        triggers = [
            EmotionTrigger(text="happy", start=0, end=5),
            EmotionTrigger(text="joyful", start=20, end=26),
        ]
        merged = _merge_triggers(triggers)
        assert len(merged) == 2


class TestEmotionConfig:
    def test_defaults(self):
        cfg = EmotionConfig()
        assert cfg.model == "nrc"
        assert cfg.return_spans is True
        assert cfg.min_words_for_emotion == 1
        assert cfg.dominant_top_k == 1

    def test_custom_values(self):
        cfg = EmotionConfig(
            model="distilroberta",
            return_spans=False,
            min_words_for_emotion=2,
            dominant_top_k=3,
            device="cuda",
            batch_size=32,
        )
        assert cfg.model == "distilroberta"
        assert cfg.return_spans is False
        assert cfg.min_words_for_emotion == 2
        assert cfg.dominant_top_k == 3
        assert cfg.device == "cuda"
        assert cfg.batch_size == 32


class TestRenPyMapping:
    def test_canonical_to_renpy(self):
        assert map_to_renpy_expression("neutral") == "normal"
        assert map_to_renpy_expression("joy") == "smile"
        assert map_to_renpy_expression("sadness") == "sad"
        assert map_to_renpy_expression("anger") == "angry"
        assert map_to_renpy_expression("fear") == "scared"
        assert map_to_renpy_expression("surprise") == "surprised"
        assert map_to_renpy_expression("disgust") == "disgusted"
        assert map_to_renpy_expression("unknown") == "normal"


class TestIntegration:
    @pytest.mark.slow
    def test_tag_emotions_nrc(self):
        text = (
            "Chapter 1\n\n"
            "He laughed with joy. \"I'm so happy!\"\n\n"
            "She cried in sadness. \"This is terrible,\" she said angrily.\n\n"
            "A shadow appeared, causing fear and surprise."
        )
        chapters, lines = preprocess(text, is_text=True)
        line_types = classify_lines(lines)
        cfg = EmotionConfig(model="nrc", return_spans=True)
        emotions = tag_emotions(lines, line_types, cfg)

        # Check we get emotions for each line
        assert len(emotions) == len(lines)
        for line_emos in emotions:
            assert len(line_emos) >= 1
            assert all(isinstance(e, EmotionTag) for e in line_emos)

    @pytest.mark.slow
    def test_tag_emotions_with_spans(self):
        text = "He laughed happily."
        chapters, lines = preprocess(text, is_text=True)
        line_types = classify_lines(lines)
        cfg = EmotionConfig(model="nrc", return_spans=True)
        emotions = tag_emotions(lines, line_types, cfg)

        assert len(emotions) == 1
        for e in emotions[0]:
            if e.trigger:
                assert isinstance(e.trigger, EmotionTrigger)
                assert e.trigger.start >= 0
                assert e.trigger.end > e.trigger.start
                assert e.trigger.text in "He laughed happily."

    @pytest.mark.slow
    def test_tag_emotions_simple(self):
        text = "He was happy."
        chapters, lines = preprocess(text, is_text=True)
        line_types = classify_lines(lines)
        cfg = EmotionConfig(model="nrc")
        simple = tag_emotions_simple(lines, line_types, cfg)

        assert len(simple) == 1
        assert isinstance(simple[0], EmotionTag)
        # "happy" in NRC has joy, anticipation, positive, trust
        # Canonical: joy, surprise, joy, joy -> deduped to {joy, surprise}
        # With priority surprise(7) > joy(5), surprise wins
        assert simple[0].label in ["joy", "surprise", "neutral"]

    @pytest.mark.slow
    def test_fear_priority_over_anger(self):
        # "fear" in NRC has anger, fear, negative
        # Should pick fear due to higher priority
        text = "He felt fear."
        chapters, lines = preprocess(text, is_text=True)
        line_types = classify_lines(lines)
        cfg = EmotionConfig(model="nrc", return_spans=True)
        emotions = tag_emotions(lines, line_types, cfg)

        assert emotions[0][0].label == "fear"

    @pytest.mark.slow
    def test_angrily_stemmed(self):
        # "angrily" -> "angry" via ily stemming
        text = "He said angrily."
        chapters, lines = preprocess(text, is_text=True)
        line_types = classify_lines(lines)
        cfg = EmotionConfig(model="nrc", return_spans=True)
        emotions = tag_emotions(lines, line_types, cfg)

        # Should detect anger from "angrily"
        found_anger = any(e.label == "anger" for e in emotions[0])
        assert found_anger, f"Expected anger, got {[e.label for e in emotions[0]]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])