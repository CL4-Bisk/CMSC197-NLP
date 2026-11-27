"""Pipeline 1, Step 7: Scene/Event Segmentation.

Segments classified lines into Scenarios (setting-based) and PlotEvents (narrative beats).
Uses embedding similarity drops, weak labels (***), POV tracking, character/time/setting signals.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from src.converter.preprocess import RawLine
from src.converter.classifier import LineType
from src.converter.characters import CharacterEnrichedLine
from src.schema import (
    PlotEvent, Scenario, ScriptLine, SourceRef, TimeOfDay, CharacterEntry
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SegmenterConfig:
    method: Literal["embedding", "texttiling", "hybrid"] = "embedding"
    window_size: int = 10              # sentences for embedding window
    threshold: float = 0.5             # boundary score threshold
    use_weak_labels: bool = True       # hard boundaries at *** breaks
    min_segment_length: int = 3        # minimum lines per scenario
    time_detection: Literal["rules", "dateparser"] = "rules"
    setting_prototypes: str = "auto"   # "auto" or path to custom YAML

    # POV tracking
    pov_tracking: bool = True
    min_pov_run_lines: int = 5
    min_pov_run_ratio: float = 0.2
    pov_persistence_window: int = 5


# --------------------------------------------------------------------------- #
# Internal data structures
# --------------------------------------------------------------------------- #

@dataclass
class SegmentationSignal:
    """Signals at each line boundary (between line i and i+1)."""
    index: int                           # boundary after this line index
    embedding_drop: float = 0.0          # cosine similarity drop
    depth_score: float = 0.0             # Hearst depth score
    scene_break: bool = False            # *** / --- / ###
    setting_change: float = 0.0          # setting prototype shift
    character_jaccard: float = 1.0       # character set overlap (1 = same)
    time_jump: float = 0.0               # time phrase detected
    pov_shift: bool = False              # POV character changed
    pov_run_length: int = 0              # length of new POV run


@dataclass
class ScenarioDraft:
    """Draft scenario before PlotEvent grouping."""
    lines: list[CharacterEnrichedLine]
    setting_name: str
    time_of_day: Optional[TimeOfDay]
    start_idx: int
    end_idx: int


# --------------------------------------------------------------------------- #
# Setting prototypes (xianxia/xuanhuan + general)
# --------------------------------------------------------------------------- #

DEFAULT_SETTING_PROTOTYPES = [
    # Cultivation settings
    "sect courtyard", "training ground", "alchemy lab", "spirit beast forest",
    "ancient ruins", "secret realm", "cultivation cave", "meditation chamber",
    "auction house", "pill hall", "scripture pavilion", "disciplinary hall",
    "grand hall", "throne room", "palace garden", "imperial palace",
    # General settings
    "forest at night", "forest day", "mountain path", "city street",
    "restaurant", "inn bedroom", "tavern", "market", "battlefield",
    "cave", "river", "lake", "cliff", "valley", "desert", "snowfield",
    # Indoor
    "bedroom", "study", "meeting room", "dining hall", "library",
    "laboratory", "workshop", "prison cell", "dungeon",
]


# --------------------------------------------------------------------------- #
# Embedding model (lazy load)
# --------------------------------------------------------------------------- #

_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# --------------------------------------------------------------------------- #
# POV Tracking
# --------------------------------------------------------------------------- #

def _infer_pov_character(
    line: CharacterEnrichedLine,
    enriched_lines: list[CharacterEnrichedLine],
    index: int,
    window: int,
    characters: list[CharacterEntry],
) -> Optional[str]:
    """Infer POV character for a line."""
    # Monologue = that character's internal POV
    if line.line_type == LineType.MONOLOGUE and line.speaker_id:
        return line.speaker_id

    # Dialogue speaker = their POV when speaking
    if line.line_type == LineType.DIALOGUE and line.speaker_id:
        return line.speaker_id

    # Narration: most mentioned character in recent window
    if line.line_type == LineType.NARRATION:
        char_counts = {}
        start = max(0, index - window)
        for i in range(start, index + 1):
            if i >= len(enriched_lines):
                break
            l = enriched_lines[i]
            if l.speaker_id:
                char_counts[l.speaker_id] = char_counts.get(l.speaker_id, 0) + 1
            # Check narration for character names
            for char in characters:
                for alias in char.aliases:
                    if alias.lower() in l.text.lower():
                        char_counts[char.id] = char_counts.get(char.id, 0) + 1
        if char_counts:
            return max(char_counts, key=char_counts.get)

    return None


def _track_pov_shifts(
    enriched_lines: list[CharacterEnrichedLine],
    characters: list[CharacterEntry],
    config: SegmenterConfig,
) -> list[Optional[str]]:
    """Track POV character per line."""
    pov_per_line = []
    for i, line in enumerate(enriched_lines):
        pov = _infer_pov_character(line, enriched_lines, i, config.pov_persistence_window, characters)
        pov_per_line.append(pov)
    return pov_per_line


# --------------------------------------------------------------------------- #
# Time Detection
# --------------------------------------------------------------------------- #

TIME_PHRASES = {
    TimeOfDay.MORNING: [
        "dawn", "sunrise", "early morning", "morning light", "first light",
        "break of day", "crack of dawn", "dewy morning", "morning mist",
    ],
    TimeOfDay.DAY: [
        "midday", "noon", "daytime", "broad daylight", "high sun",
        "sun high", "afternoon sun",
    ],
    TimeOfDay.AFTERNOON: [
        "afternoon", "late afternoon", "mid-afternoon", "waning afternoon",
    ],
    TimeOfDay.SUNSET: [
        "sunset", "dusk", "twilight", "evening", "sun set", "golden hour",
        "fading light", "dim light", "nightfall approaches",
    ],
    TimeOfDay.NIGHT: [
        "night", "midnight", "darkness", "pitch black", "moonlight",
        "starlight", "dead of night", "late night", "deep night",
    ],
}


def _detect_time_phrases(text: str) -> Optional[TimeOfDay]:
    """Rule-based time phrase detection."""
    text_lower = text.lower()
    scores = {tod: 0 for tod in TimeOfDay}
    for tod, phrases in TIME_PHRASES.items():
        for phrase in phrases:
            if phrase in text_lower:
                scores[tod] += 1
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return None


def _detect_time_dateparser(text: str) -> Optional[TimeOfDay]:
    """dateparser-based time detection (optional)."""
    try:
        import dateparser
        dt = dateparser.parse(text, settings={"PREFER_DATES_FROM": "past"})
        if dt:
            hour = dt.hour
            if 5 <= hour < 12:
                return TimeOfDay.MORNING
            elif 12 <= hour < 17:
                return TimeOfDay.AFTERNOON
            elif 17 <= hour < 21:
                return TimeOfDay.SUNSET
            else:
                return TimeOfDay.NIGHT
    except Exception:
        pass
    return None


def _detect_time_of_day(text: str, method: str) -> Optional[TimeOfDay]:
    if method == "dateparser":
        return _detect_time_dateparser(text)
    return _detect_time_phrases(text)


# --------------------------------------------------------------------------- #
# Setting Detection
# --------------------------------------------------------------------------- #

_setting_embeddings = None


def _get_setting_embeddings():
    global _setting_embeddings
    if _setting_embeddings is None:
        model = _get_embedding_model()
        _setting_embeddings = model.encode(DEFAULT_SETTING_PROTOTYPES)
    return _setting_embeddings


def _detect_setting(text: str) -> str:
    """Match text to nearest setting prototype."""
    model = _get_embedding_model()
    setting_embeds = _get_setting_embeddings()
    text_embed = model.encode([text])[0]

    sims = np.dot(setting_embeds, text_embed) / (
        np.linalg.norm(setting_embeds, axis=1) * np.linalg.norm(text_embed) + 1e-8
    )
    best_idx = int(np.argmax(sims))
    return DEFAULT_SETTING_PROTOTYPES[best_idx]


# --------------------------------------------------------------------------- #
# Character Jaccard
# --------------------------------------------------------------------------- #

def _character_jaccard(line1: CharacterEnrichedLine, line2: CharacterEnrichedLine) -> float:
    """Jaccard similarity of character sets in recent window."""
    # Get characters mentioned in each line
    chars1 = set()
    chars2 = set()
    if line1.speaker_id:
        chars1.add(line1.speaker_id)
    if line2.speaker_id:
        chars2.add(line2.speaker_id)
    # Check narration for names (simplified)
    return len(chars1 & chars2) / len(chars1 | chars2) if chars1 or chars2 else 1.0


# --------------------------------------------------------------------------- #
# Embedding Similarity
# --------------------------------------------------------------------------- #

def _compute_embedding_signals(
    enriched_lines: list[CharacterEnrichedLine],
    window_size: int,
) -> list[SegmentationSignal]:
    """Compute embedding drop and depth score at each boundary."""
    model = _get_embedding_model()
    texts = [line.text for line in enriched_lines]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    signals = []
    n = len(enriched_lines)

    for i in range(n - 1):
        # Cosine similarity between adjacent lines
        a, b = embeddings[i], embeddings[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        drop = 1.0 - cos_sim

        # Depth score (Hearst): average similarity to left window vs right window
        left_start = max(0, i - window_size + 1)
        right_end = min(n, i + window_size + 1)

        left_sims = [np.dot(a, embeddings[j]) / (np.linalg.norm(a) * np.linalg.norm(embeddings[j]) + 1e-8)
                     for j in range(left_start, i + 1)]
        right_sims = [np.dot(b, embeddings[j]) / (np.linalg.norm(b) * np.linalg.norm(embeddings[j]) + 1e-8)
                      for j in range(i + 1, right_end)]

        left_avg = np.mean(left_sims) if left_sims else 1.0
        right_avg = np.mean(right_sims) if right_sims else 1.0
        depth = (left_avg + right_avg) / 2 - cos_sim

        signals.append(SegmentationSignal(
            index=i,
            embedding_drop=drop,
            depth_score=depth,
        ))

    # Add dummy signal at end
    signals.append(SegmentationSignal(index=n - 1))
    return signals


# --------------------------------------------------------------------------- #
# Combine Signals
# --------------------------------------------------------------------------- #

def _combine_signals(
    enriched_lines: list[CharacterEnrichedLine],
    characters: list[CharacterEntry],
    config: SegmenterConfig,
) -> list[SegmentationSignal]:
    """Combine all signals into unified boundary scores."""
    n = len(enriched_lines)

    # 1. Embedding signals
    signals = _compute_embedding_signals(enriched_lines, config.window_size)

    # 2. Weak labels (scene breaks)
    if config.use_weak_labels:
        for i, line in enumerate(enriched_lines[:-1]):
            if line.scene_break_before:
                signals[i].scene_break = True

    # 3. Setting change
    for i in range(n - 1):
        setting1 = _detect_setting(enriched_lines[i].text)
        setting2 = _detect_setting(enriched_lines[i + 1].text)
        if setting1 != setting2:
            signals[i].setting_change = 1.0

    # 4. Character Jaccard
    for i in range(n - 1):
        signals[i].character_jaccard = _character_jaccard(enriched_lines[i], enriched_lines[i + 1])

    # 5. Time jump
    for i in range(n - 1):
        time1 = _detect_time_of_day(enriched_lines[i].text, config.time_detection)
        time2 = _detect_time_of_day(enriched_lines[i + 1].text, config.time_detection)
        if time1 and time2 and time1 != time2:
            signals[i].time_jump = 1.0

    # 6. POV tracking
    if config.pov_tracking:
        pov_per_line = _track_pov_shifts(enriched_lines, characters, config)
        for i in range(n - 1):
            if pov_per_line[i] != pov_per_line[i + 1]:
                signals[i].pov_shift = True
                # Track run length of new POV
                new_pov = pov_per_line[i + 1]
                run_len = 1
                for j in range(i + 2, n):
                    if pov_per_line[j] == new_pov:
                        run_len += 1
                    else:
                        break
                signals[i].pov_run_length = run_len

    return signals


# --------------------------------------------------------------------------- #
# Boundary Detection
# --------------------------------------------------------------------------- #

def _score_boundary(signal: SegmentationSignal, config: SegmenterConfig) -> float:
    """Compute unified boundary score (0-1)."""
    score = 0.0

    # Hard boundary
    if signal.scene_break:
        return 1.0

    # Weighted signals
    score += signal.embedding_drop * 0.3
    score += signal.depth_score * 0.2
    score += signal.setting_change * 0.25
    score += (1.0 - signal.character_jaccard) * 0.15
    score += signal.time_jump * 0.1

    # POV shift with volume threshold
    if signal.pov_shift:
        if signal.pov_run_length >= config.min_pov_run_lines:
            score += 0.3
        # Could also check ratio against current segment length

    return min(score, 1.0)


def _find_boundaries(
    signals: list[SegmentationSignal],
    config: SegmenterConfig,
) -> list[int]:
    """Find boundary indices where score >= threshold."""
    boundaries = [0]  # Always start at 0
    for i, signal in enumerate(signals[:-1]):  # Exclude dummy at end
        score = _score_boundary(signal, config)
        if score >= config.threshold:
            # Ensure minimum segment length (but always respect hard scene breaks)
            if signal.scene_break or i - boundaries[-1] >= config.min_segment_length:
                boundaries.append(i + 1)
    boundaries.append(len(signals))  # End
    return boundaries


# --------------------------------------------------------------------------- #
# Build Scenarios
# --------------------------------------------------------------------------- #

def _build_scenarios(
    enriched_lines: list[CharacterEnrichedLine],
    boundaries: list[int],
) -> list[ScenarioDraft]:
    """Build ScenarioDraft objects from boundaries."""
    drafts = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        segment_lines = enriched_lines[start:end]

        # Detect setting for segment
        segment_text = " ".join(l.text for l in segment_lines)
        setting_name = _detect_setting(segment_text)

        # Detect time for segment
        time_of_day = _detect_time_of_day(segment_text, "rules")

        drafts.append(ScenarioDraft(
            lines=segment_lines,
            setting_name=setting_name,
            time_of_day=time_of_day,
            start_idx=start,
            end_idx=end,
        ))
    return drafts


# --------------------------------------------------------------------------- #
# Group Scenarios into PlotEvents
# --------------------------------------------------------------------------- #

def _group_into_events(
    scenarios: list[ScenarioDraft],
    characters: list[CharacterEntry],
) -> list[PlotEvent]:
    """Group scenarios into PlotEvents using character/setting/time continuity."""
    if not scenarios:
        return []

    events = []
    current_event_scenarios = [scenarios[0]]
    event_chapter = 1  # Will be set from first scenario's lines

    for i in range(1, len(scenarios)):
        prev = scenarios[i - 1]
        curr = scenarios[i]

        # Continuity signals
        char_overlap = _scenario_character_jaccard(prev, curr)
        same_setting = prev.setting_name == curr.setting_name
        same_time = prev.time_of_day == curr.time_of_day
        time_continuous = _time_continuous(prev.time_of_day, curr.time_of_day)

        # Split if major discontinuity
        split = False
        if not same_setting:
            split = True
        elif not time_continuous:
            split = True
        elif char_overlap < 0.3:
            split = True

        if split:
            # Finalize current event
            events.append(_make_event(current_event_scenarios, len(events) + 1))
            current_event_scenarios = [curr]
        else:
            current_event_scenarios.append(curr)

    # Final event
    if current_event_scenarios:
        events.append(_make_event(current_event_scenarios, len(events) + 1))

    return events


def _scenario_character_jaccard(s1: ScenarioDraft, s2: ScenarioDraft) -> float:
    """Character Jaccard between two scenarios."""
    chars1 = set()
    chars2 = set()
    for line in s1.lines:
        if line.speaker_id:
            chars1.add(line.speaker_id)
    for line in s2.lines:
        if line.speaker_id:
            chars2.add(line.speaker_id)
    if not chars1 and not chars2:
        return 1.0
    return len(chars1 & chars2) / len(chars1 | chars2)


def _time_continuous(t1: Optional[TimeOfDay], t2: Optional[TimeOfDay]) -> bool:
    if not t1 or not t2:
        return True  # Unknown = assume continuous
    if t1 == t2:
        return True
    # Adjacent times are continuous
    order = [TimeOfDay.MORNING, TimeOfDay.DAY, TimeOfDay.AFTERNOON,
             TimeOfDay.SUNSET, TimeOfDay.NIGHT]
    try:
        i1, i2 = order.index(t1), order.index(t2)
        return abs(i1 - i2) <= 1
    except ValueError:
        return True


def _make_event(scenarios: list[ScenarioDraft], event_num: int) -> PlotEvent:
    """Create PlotEvent from scenario drafts."""
    all_lines = []
    for scn in scenarios:
        all_lines.extend(scn.lines)

    # Determine chapter from first line
    chapter = all_lines[0].chapter if all_lines else 1

    # Build Scenario objects
    schema_scenarios = []
    for i, scn in enumerate(scenarios):
        schema_scenarios.append(Scenario(
            id=f"scn_{scn.start_idx}_{i:03d}",
            name=scn.setting_name,
            time=scn.time_of_day,
            background=None,
            lines=[
                ScriptLine(
                    id=f"ln_{line.chapter}_{line.paragraph}_{j:03d}",
                    text=line.text,
                    type=line.line_type,
                    speaker=line.speaker_id,
                    emotions=getattr(line, 'emotions', []),
                    source_ref=line.source_ref(),
                )
                for j, line in enumerate(scn.lines)
            ],
        ))

    # Character IDs present
    char_ids = set()
    for line in all_lines:
        if line.speaker_id:
            char_ids.add(line.speaker_id)

    return PlotEvent(
        id=f"evt_{event_num:03d}",
        chapter=chapter,
        summary=None,
        label=None,
        characters=list(char_ids),
        scenarios=schema_scenarios,
    )


# --------------------------------------------------------------------------- #
# Main Pipeline Function
# --------------------------------------------------------------------------- #

def segment_events(
    enriched_lines: list[CharacterEnrichedLine],
    characters: list[CharacterEntry],
    config: SegmenterConfig | None = None,
) -> list[PlotEvent]:
    """
    Main segmentation pipeline.

    Args:
        enriched_lines: CharacterEnrichedLine with speaker_id and emotions
        characters: CharacterEntry list from character extraction
        config: SegmenterConfig

    Returns:
        list[PlotEvent] with Scenario children and ScriptLine leaves
    """
    config = config or SegmenterConfig()

    # 1. Combine all signals
    signals = _combine_signals(enriched_lines, characters, config)

    # 2. Find boundaries
    boundaries = _find_boundaries(signals, config)

    # 3. Build scenarios
    scenario_drafts = _build_scenarios(enriched_lines, boundaries)

    # 4. Group into PlotEvents
    events = _group_into_events(scenario_drafts, characters)

    return events


if __name__ == "__main__":
    # Quick test
    from src.converter.preprocess import preprocess
    from src.converter.classifier import classify_lines
    from src.converter.characters import extract_characters, CharacterConfig
    from src.converter.emotions import tag_emotions, EmotionConfig

    text = (
        "Chapter 1\n\n"
        "Lin Feng stared at the blade. \"I won't lose,\" he said angrily.\n\n"
        "*Why is my heart racing with fear?*\n\n"
        "The elder nodded. \"Good.\"\n\n"
        "***\n\n"
        "The forest was dark. Lin Feng walked carefully.\n\n"
        "A shadow emerged. \"Who are you?\" it asked."
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

    print(f"Events: {len(events)}")
    for evt in events:
        print(f"  Event {evt.id} (ch{evt.chapter}): {len(evt.scenarios)} scenarios")
        for scn in evt.scenarios:
            print(f"    Scenario {scn.id}: {scn.name} ({scn.time}) - {len(scn.lines)} lines")