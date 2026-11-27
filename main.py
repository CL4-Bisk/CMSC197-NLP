#!/usr/bin/env python3
"""Pipeline 1 (Converter): Web Novel → Structured VN Script.

Usage:
    python main.py path/to/novel.txt          # Process single file
    python main.py path/to/novel_folder/      # Process folder (one file per chapter)
    python main.py path/to/novel.txt -o out.json  # Specify output
    python main.py --config configs/pipeline1.yaml path/to/novel.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.converter.preprocess import preprocess, PreprocessConfig
from src.converter.classifier import classify_lines, ClassifierConfig, ClassifierModel
from src.converter.characters import (
    extract_characters,
    CharacterConfig,
    to_character_entries,
)
from src.converter.emotions import tag_emotions, EmotionConfig
from src.converter.segmenter import segment_events, SegmenterConfig
from src.schema import (
    VNScript,
    ScriptMeta,
    SourceType,
    PipelineType,
    LineType,
    ScriptLine,
    SourceRef,
    PlotEvent,
    Scenario,
    TimeOfDay,
)


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_preprocess_config(cfg: dict) -> PreprocessConfig:
    pcfg = cfg.get("preprocess", {})
    return PreprocessConfig(
        single_newline_paragraphs=pcfg.get("single_newline_paragraphs", False),
        drop_noise=pcfg.get("drop_noise", True),
        extra_noise_patterns=tuple(pcfg.get("extra_noise_patterns", [])),
    )


def build_classifier_config(cfg: dict) -> ClassifierConfig:
    ccfg = cfg.get("classifier", {})
    model_str = ccfg.get("model", "rules")
    model = ClassifierModel.RULES if model_str == "rules" else ClassifierModel.BART_ZEROSHOT
    return ClassifierConfig(
        model=model,
        confidence_threshold=ccfg.get("confidence_threshold", 0.7),
        device=ccfg.get("device", "auto"),
        batch_size=ccfg.get("batch_size", 16),
    )


def build_character_config(cfg: dict) -> CharacterConfig:
    chcfg = cfg.get("characters", {})
    return CharacterConfig(
        ner_model=chcfg.get("ner_model", "spacy_sm"),
        coref_model=chcfg.get("coref_model", "fastcoref"),
        fuzzy_threshold=chcfg.get("fuzzy_threshold", 85),
        title_prefixes=tuple(chcfg.get("title_prefixes", [])),
        attribution_window=chcfg.get("attribution_window", 3),
        use_alternation=chcfg.get("use_alternation", True),
    )


def build_emotion_config(cfg: dict) -> EmotionConfig:
    ecfg = cfg.get("emotions", {})
    model_str = ecfg.get("model", "nrc")
    return EmotionConfig(
        model=model_str,
        return_spans=ecfg.get("return_spans", True),
        min_words_for_emotion=ecfg.get("min_words_for_emotion", 1),
        dominant_top_k=ecfg.get("dominant_top_k", 1),
        device=ecfg.get("device", "auto"),
        batch_size=ecfg.get("batch_size", 16),
    )


def build_segmenter_config(cfg: dict) -> SegmenterConfig:
    scfg = cfg.get("segmenter", {})
    return SegmenterConfig(
        method=scfg.get("method", "embedding"),
        window_size=scfg.get("window_size", 10),
        threshold=scfg.get("threshold", 0.5),
        use_weak_labels=scfg.get("use_weak_labels", True),
        min_segment_length=scfg.get("min_segment_length", 3),
        time_detection=scfg.get("time_detection", "rules"),
        pov_tracking=scfg.get("pov_tracking", True),
        min_pov_run_lines=scfg.get("min_pov_run_lines", 5),
        min_pov_run_ratio=scfg.get("min_pov_run_ratio", 0.2),
    )


def convert_to_vnscript_from_events(
    events: list[PlotEvent],
    characters,
    source_path: str,
) -> VNScript:
    """Convert segmented PlotEvents to VNScript."""
    char_entries = to_character_entries(characters)

    # Build metadata
    meta = ScriptMeta(
        title=Path(source_path).stem,
        source_type=SourceType.WEB_NOVEL,
        pipeline=PipelineType.PIPELINE_1,
        models={
            "preprocess": "trafilatura+regex",
            "classifier": "rules",
            "characters": "spacy_sm+rapidfuzz",
            "emotions": "nrc",
            "segmenter": "embedding",
        },
    )

    return VNScript(meta=meta, characters=char_entries, events=events)


def main():
    parser = argparse.ArgumentParser(description="Pipeline 1: Web Novel → VN Script")
    parser.add_argument("source", help="Path to novel file/folder, or raw text (with --is-text)")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--config", default="configs/pipeline1.yaml", help="Config YAML path")
    parser.add_argument("--is-text", action="store_true", help="Treat source as raw text, not file path")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    preprocess_cfg = build_preprocess_config(config)
    classifier_cfg = build_classifier_config(config)
    character_cfg = build_character_config(config)

    print(f"Loading: {args.source}")
    chapters, lines = preprocess(args.source, preprocess_cfg, is_text=args.is_text)

    print(f"Classifying {len(lines)} lines...")
    line_types = classify_lines(lines, classifier_cfg)

    print("Extracting characters...")
    characters, enriched = extract_characters(chapters, lines, line_types, character_cfg)

    print(f"Found {len(characters)} characters:")
    for c in characters:
        print(f"  {c.id}: {c.name} (aliases: {len(c.aliases)})")

    print("Tagging emotions...")
    emotion_cfg = build_emotion_config(config)
    emotions = tag_emotions(lines, line_types, emotion_cfg)
    # Attach emotions to enriched lines
    for e, emos in zip(enriched, emotions):
        e.emotions = emos

    print("Segmenting events...")
    segmenter_cfg = build_segmenter_config(config)
    events = segment_events(enriched, to_character_entries(characters), segmenter_cfg)

    print(f"Found {len(events)} plot events:")
    for evt in events:
        print(f"  {evt.id} (ch{evt.chapter}): {len(evt.scenarios)} scenarios")

    print("Building VNScript...")
    script = convert_to_vnscript_from_events(events, characters, args.source)

    # Output
    output_data = script.model_dump(mode="json")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Written to {out_path}")
    else:
        json.dump(output_data, sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()