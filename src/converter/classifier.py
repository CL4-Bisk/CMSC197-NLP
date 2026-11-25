"""Pipeline 1, Step 4: Line Type Classification (narration / dialogue / monologue).

Default: rule-based (quotes -> dialogue, whole-line italic -> monologue, else narration).
Optional: facebook/bart-large-mnli zero-shot for ambiguous cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from src.converter.preprocess import RawLine
from src.schema import LineType


class ClassifierModel(str, Enum):
    RULES = "rules"
    BART_ZEROSHOT = "bart_zeroshot"


@dataclass(frozen=True)
class ClassifierConfig:
    model: ClassifierModel = ClassifierModel.RULES
    # For bart_zeroshot: minimum confidence to accept prediction; below falls back to rules
    confidence_threshold: float = 0.7
    # Device for transformers: "cpu", "cuda", "mps", or "auto"
    device: str = "auto"
    # Batch size for zero-shot inference
    batch_size: int = 16


# --------------------------------------------------------------------------- #
# Rule-based classification
# --------------------------------------------------------------------------- #

def _classify_rules(line: RawLine) -> LineType:
    """Deterministic, fast, explainable classification."""
    if line.quoted:
        return LineType.DIALOGUE
    if line.italic:
        return LineType.MONOLOGUE
    return LineType.NARRATION


def classify_lines_rules(lines: list[RawLine]) -> list[LineType]:
    """Classify all lines using rule-based logic."""
    return [_classify_rules(ln) for ln in lines]


# --------------------------------------------------------------------------- #
# Zero-shot classification (facebook/bart-large-mnli)
# --------------------------------------------------------------------------- #

_ZEROSHOT_LABELS = ["narration", "dialogue", "monologue"]
_LABEL_TO_LINETYPE = {
    "narration": LineType.NARRATION,
    "dialogue": LineType.DIALOGUE,
    "monologue": LineType.MONOLOGUE,
}

_ZEROSHOT_PROMPT = "This line from a novel is {label}."


def _get_zeroshot_classifier(device: str):
    """Lazily load the zero-shot pipeline."""
    from transformers import pipeline
    import torch

    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        hypothesis_template=_ZEROSHOT_PROMPT,
    )


def classify_lines_zeroshot(
    lines: list[RawLine],
    config: ClassifierConfig,
) -> list[LineType]:
    """Classify lines using BART zero-shot, falling back to rules on low confidence."""
    classifier = _get_zeroshot_classifier(config.device)

    texts = [ln.text for ln in lines]
    # Process in batches
    results = classifier(
        texts,
        candidate_labels=_ZEROSHOT_LABELS,
        batch_size=config.batch_size,
        multi_label=False,
    )

    out: list[LineType] = []
    for line, result in zip(lines, results):
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        if top_score >= config.confidence_threshold:
            out.append(_LABEL_TO_LINETYPE[top_label])
        else:
            # Fallback to rules for low-confidence predictions
            out.append(_classify_rules(line))
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def classify_lines(
    lines: list[RawLine],
    config: ClassifierConfig | None = None,
) -> list[LineType]:
    """Main entry point: classify lines into narration / dialogue / monologue.

    Args:
        lines: RawLine objects from preprocess.split_lines()
        config: ClassifierConfig (model, threshold, device). Defaults to rules-only.

    Returns:
        List of LineType matching input order.
    """
    config = config or ClassifierConfig()

    if config.model == ClassifierModel.RULES:
        return classify_lines_rules(lines)
    elif config.model == ClassifierModel.BART_ZEROSHOT:
        return classify_lines_zeroshot(lines, config)
    else:
        raise ValueError(f"Unknown classifier model: {config.model}")


# --------------------------------------------------------------------------- #
# Convenience: enrich RawLine with type (for downstream steps)
# --------------------------------------------------------------------------- #

from dataclasses import replace

@dataclass
class ClassifiedLine:
    """RawLine with predicted line type attached."""
    raw: RawLine
    line_type: LineType

    # Delegate common attributes
    @property
    def chapter(self) -> int:
        return self.raw.chapter

    @property
    def paragraph(self) -> int:
        return self.raw.paragraph

    @property
    def text(self) -> str:
        return self.raw.text

    @property
    def quoted(self) -> bool:
        return self.raw.quoted

    @property
    def italic(self) -> bool:
        return self.raw.italic

    @property
    def char_start(self) -> int:
        return self.raw.char_start

    @property
    def char_end(self) -> int:
        return self.raw.char_end

    @property
    def scene_break_before(self) -> bool:
        return self.raw.scene_break_before

    def source_ref(self):
        return self.raw.source_ref()


def classify_and_attach(
    lines: list[RawLine],
    config: ClassifierConfig | None = None,
) -> list[ClassifiedLine]:
    """Classify lines and return ClassifiedLine objects for pipeline chaining."""
    types = classify_lines(lines, config)
    return [ClassifiedLine(raw=ln, line_type=lt) for ln, lt in zip(lines, types)]


if __name__ == "__main__":
    # Quick manual test
    from src.converter.preprocess import preprocess

    sample = (
        "Chapter 1\n\n"
        "Lin Feng stared at the blade. “I won't lose,” he said.\n\n"
        "*Why is my heart racing?*\n\n"
        "The wind howled."
    )
    chapters, lines = preprocess(sample, is_text=True)
    typed = classify_and_attach(lines)
    for cl in typed:
        print(f"[{cl.line_type.value}] {cl.text}")