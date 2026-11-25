"""Unit tests for Pipeline 1, Step 4 (src/converter/classifier.py)."""
import pytest

from src.converter.classifier import (
    ClassifierConfig,
    ClassifierModel,
    classify_lines,
    classify_lines_rules,
    classify_and_attach,
    ClassifiedLine,
)
from src.converter.preprocess import RawLine, preprocess
from src.schema import LineType


def make_line(text: str, quoted: bool = False, italic: bool = False, **kwargs) -> RawLine:
    """Helper to create a RawLine with minimal required fields."""
    defaults = dict(
        chapter=1,
        paragraph=1,
        text=text,
        quoted=quoted,
        italic=italic,
        char_start=0,
        char_end=len(text),
    )
    defaults.update(kwargs)
    return RawLine(**defaults)


class TestRuleBasedClassification:
    """Tests for the default rule-based classifier."""

    def test_quoted_is_dialogue(self):
        lines = [make_line("Hello there.", quoted=True)]
        assert classify_lines_rules(lines) == [LineType.DIALOGUE]

    def test_whole_line_italic_is_monologue(self):
        lines = [make_line("Why me?", italic=True)]
        assert classify_lines_rules(lines) == [LineType.MONOLOGUE]

    def test_plain_text_is_narration(self):
        lines = [make_line("The sun rose over the hills.")]
        assert classify_lines_rules(lines) == [LineType.NARRATION]

    def test_quoted_overrides_italic(self):
        # Quoted + italic -> dialogue (quotes win)
        lines = [make_line("I'm thinking", quoted=True, italic=True)]
        assert classify_lines_rules(lines) == [LineType.DIALOGUE]

    def test_mixed_batch(self):
        lines = [
            make_line("Narration line."),
            make_line('"Dialogue."', quoted=True),
            make_line("*Monologue.*", italic=True),
            make_line("More narration."),
        ]
        expected = [
            LineType.NARRATION,
            LineType.DIALOGUE,
            LineType.MONOLOGUE,
            LineType.NARRATION,
        ]
        assert classify_lines_rules(lines) == expected


class TestClassifyLinesAPI:
    """Tests for the main classify_lines() entry point."""

    def test_default_config_uses_rules(self):
        lines = [make_line('"Hi."', quoted=True)]
        # No config passed -> defaults to rules
        result = classify_lines(lines)
        assert result == [LineType.DIALOGUE]

    def test_explicit_rules_config(self):
        cfg = ClassifierConfig(model=ClassifierModel.RULES)
        lines = [make_line("*Thoughts*", italic=True)]
        assert classify_lines(lines, cfg) == [LineType.MONOLOGUE]

    def test_empty_input(self):
        assert classify_lines([]) == []


class TestClassifiedLineWrapper:
    """Tests for ClassifiedLine dataclass."""

    def test_delegates_attributes(self):
        raw = make_line("Test", chapter=2, paragraph=3, quoted=True)
        cl = ClassifiedLine(raw=raw, line_type=LineType.DIALOGUE)

        assert cl.chapter == 2
        assert cl.paragraph == 3
        assert cl.text == "Test"
        assert cl.quoted is True
        assert cl.line_type == LineType.DIALOGUE

    def test_source_ref_passthrough(self):
        raw = make_line("Test", char_start=5, char_end=10)
        cl = ClassifiedLine(raw=raw, line_type=LineType.NARRATION)
        ref = cl.source_ref()
        assert ref.char_start == 5
        assert ref.char_end == 10


class TestIntegrationWithPreprocess:
    """End-to-end tests using real preprocess output."""

    def test_sample_from_preprocess(self):
        text = (
            "Chapter 1\n\n"
            "Lin Feng stared at the blade. “I won't lose,” he said.\n\n"
            "*Why is my heart racing?*\n\n"
            "The wind howled."
        )
        _, lines = preprocess(text, is_text=True)
        typed = classify_and_attach(lines)

        # Expected: narration, dialogue, narration (attribution), monologue, narration
        # Note: "he said." is NOT quoted - the quote closed at "I won't lose,"
        # so the attribution tag becomes a separate narration line per Corpus A split
        expected_types = [
            LineType.NARRATION,      # "Lin Feng stared at the blade."
            LineType.DIALOGUE,       # "I won't lose,"
            LineType.NARRATION,      # "he said." (attribution tag)
            LineType.MONOLOGUE,      # "Why is my heart racing?"
            LineType.NARRATION,      # "The wind howled."
        ]
        assert [cl.line_type for cl in typed] == expected_types

    def test_quoted_segments_split_correctly(self):
        text = '“Hello,” she said. “How are you?”'
        _, lines = preprocess(text, is_text=True)
        typed = classify_and_attach(lines)

        # Should be: dialogue, narration, dialogue
        assert [cl.line_type for cl in typed] == [
            LineType.DIALOGUE,
            LineType.NARRATION,
            LineType.DIALOGUE,
        ]
        assert [cl.text for cl in typed] == ["Hello,", "she said.", "How are you?"]


class TestClassifierConfig:
    def test_defaults(self):
        cfg = ClassifierConfig()
        assert cfg.model == ClassifierModel.RULES
        assert cfg.confidence_threshold == 0.7
        assert cfg.device == "auto"
        assert cfg.batch_size == 16

    def test_custom_values(self):
        cfg = ClassifierConfig(
            model=ClassifierModel.BART_ZEROSHOT,
            confidence_threshold=0.5,
            device="cpu",
            batch_size=8,
        )
        assert cfg.model == ClassifierModel.BART_ZEROSHOT
        assert cfg.confidence_threshold == 0.5
        assert cfg.device == "cpu"
        assert cfg.batch_size == 8


# Note: BART zero-shot tests are skipped by default (requires transformers + model download)
# Run with: pytest tests/test_classifier.py -k zeroshot --run-optional

@pytest.mark.skip(reason="Requires transformers and model download")
class TestBARTZeroShot:
    def test_zeroshot_classification(self):
        cfg = ClassifierConfig(model=ClassifierModel.BART_ZEROSHOT, device="cpu")
        lines = [
            make_line("The sun set slowly."),
            make_line('"I am here."', quoted=True),
            make_line("*I wonder...*", italic=True),
        ]
        result = classify_lines(lines, cfg)
        # Should at least not crash and return valid types
        assert all(isinstance(t, LineType) for t in result)
        assert len(result) == 3