"""Pipeline 1, Step 6: Emotion Tagging.

Approaches:
- NRC Emotion Lexicon (default): word-level, no ML deps, instant
- DistilRoBERTa (optional): j-hartmann/emotion-english-distilroberta-base, 7 Ekman labels
- GoEmotions (optional): SamLowe/roberta-base-go_emotions, 28 labels

Maps to canonical emotions from configs/emotions.yaml.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml

from src.converter.classifier import LineType
from src.converter.preprocess import RawLine
from src.schema import EmotionTag, EmotionTrigger


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EmotionConfig:
    model: Literal["nrc", "distilroberta", "go_emotions"] = "nrc"
    return_spans: bool = True
    min_words_for_emotion: int = 1
    dominant_top_k: int = 1
    device: str = "auto"
    batch_size: int = 16


# --------------------------------------------------------------------------- #
# Canonical mapping (NRC/GoEmotions -> our canonical set)
# --------------------------------------------------------------------------- #

# Load from configs/emotions.yaml
def _load_canonical_mapping() -> dict[str, str]:
    cfg_path = Path(__file__).parent.parent.parent / "configs" / "emotions.yaml"
    with open(cfg_path) as f:
        data = yaml.safe_load(f)
    # We need reverse mapping from model labels to canonical
    # For now, define explicitly
    return {
        # NRC emotions
        "anger": "anger",
        "anticipation": "surprise",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "joy",
        "negative": "sadness",
        "positive": "joy",
        "sadness": "sadness",
        "surprise": "surprise",
        "trust": "joy",
        # GoEmotions (subset - map to closest canonical)
        "admiration": "joy",
        "amusement": "joy",
        "annoyance": "anger",
        "approval": "joy",
        "caring": "joy",
        "confusion": "surprise",
        "curiosity": "surprise",
        "desire": "joy",
        "disappointment": "sadness",
        "disapproval": "anger",
        "embarrassment": "fear",
        "excitement": "joy",
        "gratitude": "joy",
        "grief": "sadness",
        "love": "joy",
        "nervousness": "fear",
        "optimism": "joy",
        "pride": "joy",
        "realization": "surprise",
        "relief": "joy",
        "remorse": "sadness",
    }


# DistilRoBERTa (j-hartmann) labels -> canonical
DISTILROBERTA_TO_CANONICAL = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
}

CANONICAL_EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]

# Priority for tiebreaking when multiple emotions have same count
# Higher = preferred
EMOTION_PRIORITY = {
    "fear": 10,
    "anger": 9,
    "disgust": 8,
    "surprise": 7,
    "sadness": 6,
    "joy": 5,
    "neutral": 1,
}


# --------------------------------------------------------------------------- #
# NRC Lexicon Implementation
# --------------------------------------------------------------------------- #

# Global lexicon cache
_nrc_lexicon: Optional[dict[str, set[str]]] = None


def _get_nrc_lexicon() -> dict[str, set[str]]:
    """Get NRC lexicon as word -> set of emotions."""
    global _nrc_lexicon
    if _nrc_lexicon is None:
        from nrclex import NRCLex
        # NRCLex.__lexicon__ is a dict[str, list[str]]; convert to set
        raw = NRCLex().__lexicon__
        _nrc_lexicon = {word: set(emos) for word, emos in raw.items()}
    return _nrc_lexicon


# --------------------------------------------------------------------------- #
# Tokenization for word-level spans
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"\b\w+(?:'\w+)?\b")


def _tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Tokenize text and return (word, start, end) tuples."""
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


# --------------------------------------------------------------------------- #
# NRC-based emotion tagging
# --------------------------------------------------------------------------- #

def _tag_emotions_nrc(
    lines: list[RawLine],
    config: EmotionConfig,
) -> list[list[EmotionTag]]:
    """Tag emotions using NRC Emotion Lexicon."""
    lexicon = _get_nrc_lexicon()  # dict[word] -> set[emotions]
    canonical_map = _load_canonical_mapping()

    all_tags = []

    for line in lines:
        word_emotions: dict[str, int] = {}  # canonical_emotion -> count
        word_triggers: dict[str, list[EmotionTrigger]] = {}

        for word, start, end in _tokenize_with_spans(line.text):
            word_lower = word.lower()
            # Direct match
            emotions = lexicon.get(word_lower, set())
            # Try stem (naive)
            if not emotions:
                for suffix in ("ing", "ed", "s", "es", "ly", "ily"):
                    if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                        stem = word_lower[:-len(suffix)]
                        # Handle "ily" -> "y" (e.g., angrily -> angry)
                        if suffix == "ily":
                            stem = word_lower[:-3] + "y"
                        emotions = lexicon.get(stem, set())
                        if emotions:
                            break

            # Map NRC emotions to canonical, deduplicate per word
            canonical_emotions_for_word = set()
            for emo in emotions:
                canonical = canonical_map.get(emo)
                if canonical:
                    canonical_emotions_for_word.add(canonical)

            for canonical in canonical_emotions_for_word:
                word_emotions[canonical] = word_emotions.get(canonical, 0) + 1
                if config.return_spans:
                    if canonical not in word_triggers:
                        word_triggers[canonical] = []
                    word_triggers[canonical].append(EmotionTrigger(
                        text=word,
                        start=start,
                        end=end,
                    ))

        # Build EmotionTag list
        if not word_emotions:
            # No emotion words found -> neutral
            all_tags.append([EmotionTag(label="neutral", score=1.0)])
            continue

        # Sort by count (desc), then by priority (desc) for tiebreaking
        sorted_emotions = sorted(
            word_emotions.items(),
            key=lambda x: (-x[1], -EMOTION_PRIORITY.get(x[0], 0))
        )
        total = sum(c for _, c in sorted_emotions)

        tags = []
        for emo, count in sorted_emotions[:config.dominant_top_k]:
            score = count / total if total > 0 else 0.0
            triggers = word_triggers.get(emo, []) if config.return_spans else []
            # Merge contiguous triggers into single span per emotion
            merged_triggers = _merge_triggers(triggers) if triggers else None
            tags.append(EmotionTag(
                label=emo,
                score=score,
                trigger=merged_triggers[0] if merged_triggers else None,
            ))
        all_tags.append(tags)

    return all_tags


def _merge_triggers(triggers: list[EmotionTrigger]) -> list[EmotionTrigger]:
    """Merge adjacent/contiguous triggers into single spans."""
    if not triggers:
        return []
    triggers = sorted(triggers, key=lambda t: t.start)
    merged = []
    current = triggers[0]
    for t in triggers[1:]:
        if t.start <= current.end + 1:  # adjacent or overlapping
            current = EmotionTrigger(
                text=current.text + " " + t.text,
                start=current.start,
                end=t.end,
            )
        else:
            merged.append(current)
            current = t
    merged.append(current)
    return merged


# --------------------------------------------------------------------------- #
# DistilRoBERTa / GoEmotions (transformers-based)
# --------------------------------------------------------------------------- #

def _load_transformers_classifier(model_name: str, device: str):
    """Lazy load transformers pipeline."""
    from transformers import pipeline
    import torch

    if device == "auto":
        device = 0 if torch.cuda.is_available() else -1

    if model_name == "distilroberta":
        model = "j-hartmann/emotion-english-distilroberta-base"
        return pipeline(
            "text-classification",
            model=model,
            device=device,
            top_k=None,
            return_all_scores=True,
        )
    elif model_name == "go_emotions":
        model = "SamLowe/roberta-base-go_emotions"
        return pipeline(
            "text-classification",
            model=model,
            device=device,
            top_k=None,
            return_all_scores=True,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _tag_emotions_transformers(
    lines: list[RawLine],
    config: EmotionConfig,
) -> list[list[EmotionTag]]:
    """Tag emotions using transformers model."""
    classifier = _load_transformers_classifier(config.model, config.device)
    canonical_map = DISTILROBERTA_TO_CANONICAL if config.model == "distilroberta" else _load_canonical_mapping()

    texts = [line.text for line in lines]
    results = classifier(texts, batch_size=config.batch_size)

    all_tags = []
    for line, result in zip(lines, results):
        # result is list of dicts: [{'label': 'joy', 'score': 0.9}, ...]
        # Map to canonical, aggregate
        canonical_scores: dict[str, float] = {}
        for item in result:
            canonical = canonical_map.get(item["label"])
            if canonical:
                canonical_scores[canonical] = max(canonical_scores.get(canonical, 0.0), item["score"])

        if not canonical_scores:
            all_tags.append([EmotionTag(label="neutral", score=1.0)])
            continue

        sorted_emotions = sorted(canonical_scores.items(), key=lambda x: -x[1])
        tags = []
        for emo, score in sorted_emotions[:config.dominant_top_k]:
            # For transformers, word-level spans need captum (not implemented here)
            trigger = None
            if config.return_spans:
                # Placeholder: would need captum integration
                pass
            tags.append(EmotionTag(label=emo, score=score, trigger=trigger))
        all_tags.append(tags)

    return all_tags


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def tag_emotions(
    lines: list[RawLine],
    line_types: list[LineType],
    config: EmotionConfig | None = None,
) -> list[list[EmotionTag]]:
    """
    Tag emotions for each line.

    Args:
        lines: RawLine objects from preprocess
        line_types: LineType from classifier (used to skip narration if desired)
        config: EmotionConfig

    Returns:
        List of EmotionTag lists, parallel to input lines.
    """
    config = config or EmotionConfig()

    if config.model == "nrc":
        return _tag_emotions_nrc(lines, config)
    elif config.model in ("distilroberta", "go_emotions"):
        return _tag_emotions_transformers(lines, config)
    else:
        raise ValueError(f"Unknown emotion model: {config.model}")


def tag_emotions_simple(
    lines: list[RawLine],
    line_types: list[LineType],
    config: EmotionConfig | None = None,
) -> list[EmotionTag]:
    """Convenience: return only dominant emotion per line (first tag)."""
    all_tags = tag_emotions(lines, line_types, config)
    return [tags[0] if tags else EmotionTag(label="neutral", score=1.0) for tags in all_tags]


# --------------------------------------------------------------------------- #
# Ren'Py expression mapping
# --------------------------------------------------------------------------- #

def map_to_renpy_expression(emotion: str) -> str:
    """Map canonical emotion to Ren'Py sprite expression tag."""
    # Reverse of configs/emotions.yaml renpy_expression_map
    canonical_to_renpy = {
        "neutral": "normal",
        "joy": "smile",
        "sadness": "sad",
        "anger": "angry",
        "fear": "scared",
        "surprise": "surprised",
        "disgust": "disgusted",
    }
    return canonical_to_renpy.get(emotion, "normal")


if __name__ == "__main__":
    # Quick test
    from src.converter.preprocess import preprocess
    from src.converter.classifier import classify_lines

    text = (
        "Chapter 1\n\n"
        "Lin Feng stared at the blade. \"I won't lose,\" he said angrily.\n\n"
        "*Why is my heart racing with fear?*\n\n"
        "The elder smiled happily."
    )
    chapters, lines = preprocess(text, is_text=True)
    line_types = classify_lines(lines)
    cfg = EmotionConfig(model="nrc", return_spans=True)
    emotions = tag_emotions(lines, line_types, cfg)

    for line, emos in zip(lines, emotions):
        print(f"[{line.text}] -> {[(e.label, e.score, e.trigger.text if e.trigger else None) for e in emos]}")