# Pipeline 1 Documentation

## Overview

Pipeline 1 (Converter) transforms raw web novel text into structured, classified lines ready for downstream processing (character extraction, emotion tagging, scene segmentation, export).

```
Raw Text → [Step 1-3] preprocess.py → [Step 4] classifier.py → Classified Lines
```

---

## Module: `src/converter/preprocess.py`

### Purpose
Steps 1–3: Extract raw text from source files, clean & split into chapters, split into Corpus A lines (paragraph breaks + quote boundaries).

### Public API

```python
from src.converter.preprocess import (
    load_raw,           # Step 1: single file → str
    load_book,          # Step 1: folder (one file per chapter) → str
    load_source,        # Step 1: auto-detect file vs folder → str
    clean_and_split_chapters,  # Step 2: str → list[Chapter]
    split_lines,        # Step 3: list[Chapter] → list[RawLine]
    preprocess,         # Steps 1-3 combined: path_or_text → (chapters, lines)
    PreprocessConfig,   # Configuration dataclass
    Chapter, Paragraph, RawLine  # Data classes
)
```

### Inputs

| Function | Input | Description |
|----------|-------|-------------|
| `load_raw(path)` | `str \| Path` | Single file: `.txt`, `.md`, `.html`, `.htm`, `.xhtml`, `.epub` |
| `load_book(dir)` | `str \| Path` | Folder with chapter files (sorted naturally: `ch_1`, `ch_2`, `ch_10`) |
| `load_source(path)` | `str \| Path` | Auto-detects: folder → `load_book`, file → `load_raw` |
| `preprocess(source, cfg, is_text=False)` | `str \| Path` | Unified entry: file, folder, or raw text (if `is_text=True`) |

### Supported Formats

| Format | Handler | Notes |
|--------|---------|-------|
| `.txt`, `.md` | Direct read (UTF-8, falls back to charset_normalizer) | BOM handled |
| `.html`, `.htm`, `.xhtml` | `trafilatura` (markdown output) | Keeps `*italic*`, `# headings`; fallback strips tags |
| `.epub` | Reads spine order, extracts HTML items | Joins with `\f` document boundary |
| Folder of above | `load_book` | Each file = chapter; files sorted naturally |

### Outputs

#### `Chapter` (from `clean_and_split_chapters`)
```python
@dataclass
class Chapter:
    index: int              # 1-based
    title: str | None       # From heading or None
    paragraphs: list[Paragraph]
```

#### `Paragraph`
```python
@dataclass
class Paragraph:
    index: int              # 1-based within chapter
    text: str               # Cleaned text, *italic* markers preserved
    scene_break_before: bool  # True if *** / --- / ### preceded this paragraph
```

#### `RawLine` (from `split_lines`)
```python
@dataclass
class RawLine:
    chapter: int
    paragraph: int
    text: str               # Marker-free, no surrounding quotes
    quoted: bool            # True if inside quote marks
    italic: bool            # True if entire line was *italic*
    char_start: int         # Offset into Paragraph.text (with * markers)
    char_end: int           # Offset into Paragraph.text (with * markers)
    scene_break_before: bool  # True only on first line of a paragraph after ***
    
    def source_ref(self) -> SourceRef  # For schema.py integration
```

### `PreprocessConfig` — Optional Parameters

```python
@dataclass(frozen=True)
class PreprocessConfig:
    single_newline_paragraphs: bool = False
        # True: treat each \n as paragraph break (common in web novels)
        # False: Corpus A default — paragraphs separated by blank line (\n\n)

    drop_noise: bool = True
        # True: remove translator notes, ads, promo lines, copyright
        # False: keep everything (debugging)

    extra_noise_patterns: tuple[str, ...] = ()
        # Additional regex patterns (case-insensitive) to treat as noise
        # Example: (r"^enjoy the chapter", r"^support us on")

    quote_pairs: dict[str, str] = DEFAULT_QUOTES
        # Custom quote pairs for non-standard quote styles
        # Default: {"“": "”", '"': '"', "「": "」", "『": "』", "«": "»"}
```

### Effects of Config on Output

| Config | Effect on Output |
|--------|------------------|
| `single_newline_paragraphs=True` | More, shorter paragraphs → more `RawLine` objects; chapter detection still works |
| `drop_noise=False` | Noise paragraphs become `Paragraph` objects → appear in `RawLine` output |
| `extra_noise_patterns` | Custom site-specific boilerplate removed before chapter/line split |
| `quote_pairs` | Changes which characters open/close quotes → changes `quoted` flag on `RawLine` |

### Example Usage

```python
from src.converter.preprocess import preprocess, PreprocessConfig

# Default (file or folder)
chapters, lines = preprocess("path/to/novel")

# Raw text
chapters, lines = preprocess(text, is_text=True)

# Custom config: one-paragraph-per-line web novel, keep promo lines
cfg = PreprocessConfig(
    single_newline_paragraphs=True,
    drop_noise=False,
    extra_noise_patterns=(r"^ads?\.com",)
)
chapters, lines = preprocess("webnovel_folder", cfg)
```

---

## Module: `src/converter/classifier.py`

### Purpose
Step 4: Classify each `RawLine` as **narration**, **dialogue**, or **monologue**.

### Public API

```python
from src.converter.classifier import (
    classify_lines,           # Main: list[RawLine] → list[LineType]
    classify_and_attach,      # Returns list[ClassifiedLine] (wrapper with .line_type)
    ClassifierConfig,         # Configuration
    ClassifierModel,          # Enum: RULES | BART_ZEROSHOT
    ClassifiedLine,           # RawLine + line_type
    LineType                  # Enum from schema: NARRATION | DIALOGUE | MONOLOGUE
)
```

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `lines` | `list[RawLine]` | Output from `preprocess.split_lines()` or `preprocess()` |
| `config` | `ClassifierConfig \| None` | Optional configuration (defaults to rules) |

### Outputs

| Function | Returns | Description |
|----------|---------|-------------|
| `classify_lines(lines, config)` | `list[LineType]` | Parallel list matching input order |
| `classify_and_attach(lines, config)` | `list[ClassifiedLine]` | Wrapper objects with `.line_type` + all `RawLine` attrs |

#### `ClassifiedLine`
```python
@dataclass
class ClassifiedLine:
    raw: RawLine
    line_type: LineType
    
    # Delegated properties (read from raw):
    chapter, paragraph, text, quoted, italic, char_start, char_end, scene_break_before
    def source_ref(self) -> SourceRef
```

### Classification Logic

#### Rule-Based (Default: `ClassifierModel.RULES`)

| Condition | LineType | Priority |
|-----------|----------|----------|
| `line.quoted == True` | `DIALOGUE` | 1 (highest) |
| `line.italic == True` | `MONOLOGUE` | 2 |
| otherwise | `NARRATION` | 3 (fallback) |

> **Note**: Quoted + italic → `DIALOGUE` (quotes win). Attribution tags like `he said.` are **not quoted** → `NARRATION`.

#### Zero-Shot BART (`ClassifierModel.BART_ZEROSHOT`)

Uses `facebook/bart-large-mnli` with hypothesis template: `"This line from a novel is {label}."`

- Runs on batches (`config.batch_size`, default 16)
- Returns highest-scoring label if `score >= confidence_threshold`
- Falls back to **rule-based** for low-confidence predictions
- Requires: `pip install transformers torch` + GPU recommended

### `ClassifierConfig` — Optional Parameters

```python
@dataclass(frozen=True)
class ClassifierConfig:
    model: ClassifierModel = ClassifierModel.RULES
        # RULES: deterministic, instant, no deps
        # BART_ZEROSHOT: ML-based, handles ambiguous cases

    confidence_threshold: float = 0.7
        # Only used for BART_ZEROSHOT
        # Below this → falls back to rules
        # Range: 0.0–1.0 (higher = stricter, more fallback)

    device: str = "auto"
        # "auto" → CUDA if available else CPU
        # "cpu", "cuda", "mps" (Apple Silicon)

    batch_size: int = 16
        # Only for BART_ZEROSHOT
        # Larger = faster throughput, more VRAM
```

### Effects of Config on Output

| Config | Effect |
|--------|--------|
| `model=RULES` (default) | 100% deterministic; no network/model loading; ~10k lines/sec on CPU |
| `model=BART_ZEROSHOT` | Better on edge cases (e.g., unquoted dialogue, narrated thoughts); needs GPU for speed; ~100–500 lines/sec on GPU |
| `confidence_threshold=0.9` | More fallback to rules → safer but less ML benefit |
| `confidence_threshold=0.5` | More ML predictions → higher recall on ambiguous lines, more errors |
| `device="cpu"` | Works without GPU but 10–50× slower than CUDA |

### Example Usage

```python
from src.converter.preprocess import preprocess
from src.converter.classifier import classify_lines, classify_and_attach, ClassifierConfig, ClassifierModel

# Default: rule-based
chapters, lines = preprocess("novel.txt")
types = classify_lines(lines)  # list[LineType]

# Pipeline-friendly: ClassifiedLine objects
typed = classify_and_attach(lines)
for cl in typed:
    print(f"[{cl.line_type.value}] ch{cl.chapter} p{cl.paragraph}: {cl.text}")

# Optional: BART zero-shot for higher accuracy on messy text
cfg = ClassifierConfig(
    model=ClassifierModel.BART_ZEROSHOT,
    confidence_threshold=0.75,
    device="cuda",          # or "auto"
    batch_size=32
)
types = classify_lines(lines, cfg)
```

---

## Data Flow Summary

```
Input Text
    │
    ▼
load_raw / load_book / load_source
    │
    ▼
clean_and_split_chapters(text, cfg)  ───► list[Chapter]
    │                                       │
    │    (Paragraph.text keeps *italic*,    │
    │     scene_break_before on Paras)      │
    ▼                                       ▼
split_lines(chapters, cfg)  ──────────► list[RawLine]
    │                                       │
    │    (quoted=True for dialogue,         │
    │     italic=True for *monologue*,      │
    │     char_start/char_end for refs)     │
    ▼                                       ▼
classify_lines(lines, cfg)  ──────────► list[LineType]
    │
    │   or
    ▼
classify_and_attach(lines, cfg)  ────► list[ClassifiedLine]
                                       │
                                       ▼
                            Ready for Step 5:
                            Character Extraction
```

---

## Quick Reference: LineType Distribution (Typical Novel)

| Type | Typical % | Source |
|------|-----------|--------|
| `NARRATION` | 40–60% | Unquoted, non-italic lines |
| `DIALOGUE` | 30–50% | Quoted segments (`line.quoted=True`) |
| `MONOLOGUE` | 1–5% | Whole-line italics (`line.italic=True`) |

*Monologue is rare because Ren'Py/web novels rarely mark internal thoughts with consistent italics.*

---

## Module: `src/converter/characters.py`

### Purpose
Step 5: Extract characters from text, resolve coreferences, merge aliases, and attribute speakers to dialogue lines.

**Pipeline:**
```
NER (spaCy) → Coreference (fastcoref*) → Alias Merging (rapidfuzz) → Speaker Attribution (rules)
```

*Coreference is optional; skipped if fastcoref unavailable due to version conflicts.*

### Public API

```python
from src.converter.characters import (
    extract_characters,        # Main: (chapters, lines, line_types) → (characters, enriched_lines)
    CharacterConfig,           # Configuration
    Character,                 # Internal: id, name, aliases, mentions
    Mention,                   # Internal: surface form + location
    CharacterEnrichedLine,     # Wrapper: RawLine + line_type + speaker_id
    to_character_entries,      # Convert to schema.CharacterEntry for export
)
```

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `chapters` | `list[Chapter]` | From `preprocess.clean_and_split_chapters()` |
| `lines` | `list[RawLine]` | From `preprocess.split_lines()` |
| `line_types` | `list[LineType]` | From `classifier.classify_lines()` |
| `config` | `CharacterConfig \| None` | Optional configuration |

### Outputs

| Function | Returns | Description |
|----------|---------|-------------|
| `extract_characters(...)` | `tuple[list[Character], list[CharacterEnrichedLine]]` | Characters + enriched lines with `speaker_id` |
| `to_character_entries(characters)` | `list[CharacterEntry]` | Schema-compatible for JSON export |

#### `Character` (internal)
```python
@dataclass
class Character:
    id: str                    # "char_001"
    name: str                  # Canonical: "Lin Feng"
    aliases: list[str]         # All surface forms: ["Lin Feng", "Young Master Lin", "he"]
    mentions: list[Mention]    # Where they appear in text
```

#### `Mention`
```python
@dataclass
class Mention:
    text: str                  # Surface form: "Young Master Lin"
    chapter: int
    paragraph: int
    line_idx: int              # Index in RawLine list
    char_start: int
    char_end: int
    is_pronoun: bool           # True for "he"/"she"/"they"
```

#### `CharacterEnrichedLine` (SRP wrapper)
```python
@dataclass
class CharacterEnrichedLine:
    raw: RawLine
    line_type: LineType
    speaker_id: str | None     # NEW: links to Character.id
    
    # Delegates all RawLine properties: chapter, paragraph, text, quoted, italic, ...
```

### Sub-Step Details

#### 5a. NER — `spaCy` (PERSON entities)
- **Default**: `en_core_web_sm` (fast, ~12 MB, CPU-friendly)
- **Optional**: `en_core_web_trf` (transformer, ~438 MB, higher recall on non-standard names)
- Runs on paragraph text, maps entities back to `RawLine` via character offsets
- Only `PERSON` / `PER` labels kept

#### 5b. Coreference — `fastcoref` (FCoref / LingMessCoref)
- Links pronouns ("he", "she") to antecedent names
- Runs per-chapter on full text
- **Note**: Often has transformers version conflicts; gracefully skipped if unavailable
- Adds pronoun `Mention` objects with `is_pronoun=True`

#### 5c. Alias Merging — `rapidfuzz` + Title Stripping
- Strips cultivation/family titles: "Young Master Lin" → "Lin", "Senior Brother Chen" → "Chen"
- Matches on normalized full name **OR** surname/family name
- Chinese name heuristic: first token (1-4 chars) = surname
- Clusters merged; longest name becomes canonical

#### 5d. Speaker Attribution — Rule-Based (Causal, Sequential)
| Priority | Method | Context |
|----------|--------|---------|
| 1 | Explicit tag: `"Lin Feng said"` / `"said Lin Feng"` | Prev narration line |
| 2 | Coref pronoun at dialogue start | "He smiled." → last speaker |
| 3 | Nearest named character in window | Backward search (default 3 lines) |
| 4 | Alternation (A/B/A/B) | Two-speaker dialogue heuristic |

### `CharacterConfig` — Optional Parameters

```python
@dataclass(frozen=True)
class CharacterConfig:
    ner_model: Literal["spacy_trf", "spacy_sm", "flair"] = "spacy_sm"
    coref_model: Literal["fastcoref", "lingmess"] = "fastcoref"
    fuzzy_threshold: int = 85
    title_prefixes: tuple[str, ...] = DEFAULT_TITLE_PREFIXES
    attribution_window: int = 3
    use_alternation: bool = True
```

| Parameter | Effect |
|-----------|--------|
| `ner_model="spacy_sm"` (default) | Fast, CPU, ~84% PERSON F1 |
| `ner_model="spacy_trf"` | Slower, needs 4GB+ RAM, ~90% F1, better on xianxia names |
| `fuzzy_threshold=85` | Stricter merging; lower = more aggressive |
| `title_prefixes` | Add/remove titles for your genre (xianxia, sci-fi, etc.) |
| `attribution_window=3` | Lines back to search for speaker tags |

### Example Usage

```python
from src.converter.preprocess import preprocess
from src.converter.classifier import classify_lines
from src.converter.characters import extract_characters, CharacterConfig, to_character_entries

# Full pipeline
chapters, lines = preprocess("novel.txt")
line_types = classify_lines(lines)

# Default config (spaCy small, fastcoref if available)
cfg = CharacterConfig()
characters, enriched = extract_characters(chapters, lines, line_types, cfg)

# Access speaker-attributed dialogue
for e in enriched:
    if e.line_type == LineType.DIALOGUE:
        speaker = e.speaker_id or "UNKNOWN"
        print(f"{speaker}: {e.text}")

# Convert for JSON export
entries = to_character_entries(characters)
# → list[CharacterEntry] compatible with schema.VNScript
```

### Custom Title Prefixes (per-novel)

```python
# Xianxia / cultivation
cfg = CharacterConfig(title_prefixes=(
    "young master", "senior brother", "junior brother",
    "sect master", "patriarch", "daoist", "venerable", "elder",
    # ... add your novel's specific titles
))

# Western fantasy
cfg = CharacterConfig(title_prefixes=(
    "lord", "lady", "sir", "king", "queen", "prince", "princess",
    "captain", "commander", "archmage", "high priest",
))
```

### Mystery Characters ("???" → Revealed Later)

The pipeline outputs separate entries for unrevealed characters:
```python
Character(id="char_007", name="???", aliases=["???", "the mysterious figure"])
Character(id="char_012", name="Lin Feng", aliases=["Lin Feng", "Young Master Lin"])
```

**Post-processing**: Proofreader merges after reveal:
```python
# In exported JSON, edit aliases:
char_012.aliases.extend(["???", "the mysterious figure"])
# Or use a merge script
```

---

## Module: `src/converter/emotions.py`

### Purpose
Step 6: Tag emotions for each line, with optional word-level triggers.

**Pipeline:**
```
NRC Lexicon (default) → Canonical mapping → Dominant emotion + word triggers
         ↓
Optional: DistilRoBERTa / GoEmotions (transformers-based)
```

### Public API

```python
from src.converter.emotions import (
    tag_emotions,          # Main: (lines, line_types) → list[list[EmotionTag]]
    tag_emotions_simple,   # Convenience: dominant emotion per line
    EmotionConfig,         # Configuration
    map_to_renpy_expression,  # Canonical → Ren'Py sprite tag
)
```

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `lines` | `list[RawLine]` | From `preprocess.split_lines()` |
| `line_types` | `list[LineType]` | From `classifier.classify_lines()` |
| `config` | `EmotionConfig \| None` | Optional configuration |

### Outputs

| Function | Returns | Description |
|----------|---------|-------------|
| `tag_emotions(...)` | `list[list[EmotionTag]]` | Parallel to input lines; each line gets list of EmotionTag |
| `tag_emotions_simple(...)` | `list[EmotionTag]` | Single dominant emotion per line |

#### `EmotionTag` (from schema)
```python
@dataclass
class EmotionTag:
    label: str                    # Canonical: neutral, joy, sadness, anger, fear, surprise, disgust
    score: float | None           # 0.0–1.0 (proportion of emotion words)
    trigger: EmotionTrigger | None  # Word-level span (if return_spans=True)
```

#### `EmotionTrigger`
```python
@dataclass
class EmotionTrigger:
    text: str                     # Matched word(s)
    start: int                    # Char offset in line text
    end: int
```

### Approaches

#### NRC Lexicon (Default: `model="nrc"`)
- **Dependencies**: `nrclex` (includes NLTK + TextBlob)
- **Speed**: Instant (~10k lines/sec on CPU)
- **Word-level spans**: ✅ Free (each word looked up in lexicon)
- **Context awareness**: ❌ Bag-of-words
- **Emotions**: 10 NRC → mapped to 7 canonical

#### DistilRoBERTa (`model="distilroberta"`)
- **Model**: `j-hartmann/emotion-english-distilroberta-base`
- **Dependencies**: `transformers`, `torch` (GPU recommended)
- **Speed**: ~100–500 lines/sec on GPU
- **Labels**: 7 Ekman (direct match to canonical)
- **Word spans**: Requires `captum` (not yet implemented)

#### GoEmotions (`model="go_emotions"`)
- **Model**: `SamLowe/roberta-base-go_emotions`
- **Labels**: 28 → mapped to 7 canonical

### Canonical Emotion Mapping

| NRC / GoEmotions | Canonical | Priority |
|------------------|-----------|----------|
| joy, positive, trust, admiration, amusement, approval, caring, desire, excitement, gratitude, love, optimism, pride, relief | **joy** | 5 |
| sadness, negative, disappointment, grief, remorse | **sadness** | 6 |
| anger, annoyance, disapproval | **anger** | 9 |
| fear, nervousness, embarrassment | **fear** | 10 |
| surprise, anticipation, curiosity, confusion, realization | **surprise** | 7 |
| disgust, abhorrence | **disgust** | 8 |
| neutral | **neutral** | 1 |

*Priority used for tiebreaking when multiple emotions have equal word counts.*

### `EmotionConfig` — Optional Parameters

```python
@dataclass(frozen=True)
class EmotionConfig:
    model: Literal["nrc", "distilroberta", "go_emotions"] = "nrc"
    return_spans: bool = True
    min_words_for_emotion: int = 1
    dominant_top_k: int = 1
    device: str = "auto"
    batch_size: int = 16
```

| Parameter | Effect |
|-----------|--------|
| `model="nrc"` (default) | Fast, no GPU, word spans free |
| `model="distilroberta"` | Context-aware, needs GPU, no word spans yet |
| `return_spans=True` | Include `EmotionTrigger` with word offsets |
| `dominant_top_k=1` | Number of top emotions per line (1 = single dominant) |
| `min_words_for_emotion=1` | Minimum NRC words to assign non-neutral |

### Example Usage

```python
from src.converter.preprocess import preprocess
from src.converter.classifier import classify_lines
from src.converter.emotions import tag_emotions, EmotionConfig

chapters, lines = preprocess("novel.txt")
line_types = classify_lines(lines)

# Default: NRC with word spans
cfg = EmotionConfig(model="nrc", return_spans=True, dominant_top_k=2)
emotions = tag_emotions(lines, line_types, cfg)

for line, emos in zip(lines, emotions):
    for e in emos:
        trigger = e.trigger.text if e.trigger else "line-level"
        print(f"[{e.label} {e.score:.2f}] {line.text} → trigger: {trigger}")

# Simple: dominant only
simple = tag_emotions_simple(lines, line_types, cfg)
for line, emo in zip(lines, simple):
    print(f"[{emo.label}] {line.text}")
```

### Ren'Py Expression Mapping

```python
from src.converter.emotions import map_to_renpy_expression

map_to_renpy_expression("joy")       # "smile"
map_to_renpy_expression("sadness")   # "sad"
map_to_renpy_expression("anger")     # "angry"
map_to_renpy_expression("fear")      # "scared"
map_to_renpy_expression("surprise")  # "surprised"
map_to_renpy_expression("disgust")   # "disgusted"
map_to_renpy_expression("neutral")   # "normal"
```

*Matches `configs/emotions.yaml` `renpy_expression_map`.*

### Configuration via `configs/pipeline1.yaml`

```yaml
emotions:
  model: "nrc"                    # "nrc" | "distilroberta" | "go_emotions"
  return_spans: true
  min_words_for_emotion: 1
  dominant_top_k: 1
  device: "auto"
  batch_size: 16

segmenter:
  method: "embedding"            # "texttiling" | "embedding" | "hybrid"
  window_size: 10
  threshold: 0.5
  use_weak_labels: true       # respect *** breaks
  min_segment_length: 3
  time_detection: "rules"        # "rules" | "dateparser"
  pov_tracking: true
  min_pov_run_lines: 5
  min_pov_run_ratio: 0.2
```

---

## Module: `src/converter/segmenter.py`

### Purpose
Step 7: Segment classified, character-enriched lines into **Scenarios** (setting-based segments) and **PlotEvents** (narrative beats spanning multiple scenarios).

**Pipeline:**
```
Embedding signals + Weak labels + POV tracking + Character/Time/Setting signals
    → Boundary detection
    → Scenario construction (setting + time)
    → PlotEvent grouping (continuity across scenarios)
```

### Public API

```python
from src.converter.segmenter import (
    segment_events,          # Main: (enriched_lines, characters) → list[PlotEvent]
    SegmenterConfig,         # Configuration
    SegmentationSignal,      # Internal: boundary signals
)
```

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `enriched_lines` | `list[CharacterEnrichedLine]` | From `characters.extract_characters()` (includes `speaker_id`, `emotions`) |
| `characters` | `list[CharacterEntry]` | From `characters.to_character_entries()` |
| `config` | `SegmenterConfig \| None` | Optional configuration |

### Outputs

| Function | Returns | Description |
|----------|---------|-------------|
| `segment_events(...)` | `list[PlotEvent]` | Schema-compatible PlotEvents with Scenario children |

### Segmentation Signals

| Signal | Source | Weight |
|--------|--------|--------|
| **Embedding drop** | Cosine similarity drop between adjacent lines (`all-MiniLM-L6-v2`) | 0.30 |
| **Depth score** | Hearst (1997) depth: left/right window coherence | 0.20 |
| **Scene break** | `***`, `---`, `###` from preprocess (`scene_break_before`) | **Hard** (1.0) |
| **Setting change** | Setting prototype shift (embedding nearest) | 0.25 |
| **Character Jaccard** | Speaker set overlap between adjacent lines | 0.15 |
| **Time jump** | Time phrase change (morning→night, etc.) | 0.10 |
| **POV shift** | POV character change with volume threshold | 0.30 (if run ≥ threshold) |

### Scenario Construction

Each Scenario gets:
- `name`: Nearest setting prototype (e.g., "forest at night", "sect courtyard")
- `time`: `TimeOfDay` from phrase detection ("dawn", "dusk", etc.) or `dateparser`
- `background`: Asset name placeholder (for Ren'Py)
- `lines`: `ScriptLine` objects with speaker, emotions, source_ref

### PlotEvent Grouping

Scenarios grouped into PlotEvents by continuity:
- **Same setting** → same event
- **Adjacent time** (morning→day) → same event
- **Character overlap** (Jaccard ≥ 0.3) → same event
- **No "time jump" phrases** ("three days later") → same event

Chapter boundaries are **soft signals** — events can span chapters.

### POV Tracking

| POV Source | Inference |
|------------|-----------|
| **Monologue** (`italic=True`) | That character's internal POV |
| **Dialogue speaker** | Speaker's POV when speaking |
| **Narration focus** | Most-mentioned character in window (default 5 lines) |

**Volume threshold**: POV shift only creates scenario boundary if new POV run ≥ `min_pov_run_lines` (default 5) OR ≥ `min_pov_run_ratio` (default 20% of current scenario).

### Setting Prototypes (Default)

30+ xianxia/general settings: "sect courtyard", "training ground", "alchemy lab", "spirit beast forest", "ancient ruins", "secret realm", "forest at night", "forest day", "mountain path", "city street", "restaurant", "inn bedroom", "tavern", "market", "battlefield", "cave", "river", "bedroom", "study", "meeting room", "dining hall", "library", "laboratory", "prison cell", "dungeon".

Customizable via `setting_prototypes` config (path to YAML).

### Time Detection

| Method | Description |
|--------|-------------|
| **rules** (default) | Phrase matching: "dawn"→morning, "dusk"→sunset, "midnight"→night |
| **dateparser** | Optional: parses absolute/relative dates (needs `dateparser` dep) |

### `SegmenterConfig` — Optional Parameters

```python
@dataclass(frozen=True)
class SegmenterConfig:
    method: Literal["embedding", "texttiling", "hybrid"] = "embedding"
    window_size: int = 10
    threshold: float = 0.5
    use_weak_labels: bool = True
    min_segment_length: int = 3
    time_detection: Literal["rules", "dateparser"] = "rules"
    pov_tracking: bool = True
    min_pov_run_lines: int = 5
    min_pov_run_ratio: float = 0.2
```

| Parameter | Effect |
|-----------|--------|
| `method="embedding"` | Embedding similarity + depth score (recommended) |
| `method="texttiling"` | NLTK TextTilingTokenizer (no ML deps) |
| `threshold=0.5` | Boundary score threshold (0-1) |
| `use_weak_labels=True` | Hard boundaries at `***`/`---`/`###` |
| `min_segment_length=3` | Minimum lines per scenario |
| `time_detection="rules"` | Phrase-based (fast) or "dateparser" |
| `pov_tracking=True` | Enable POV-based splitting |
| `min_pov_run_lines=5` | Lines for POV to become scenario |
| `min_pov_run_ratio=0.2` | Or % of current scenario length |

### Example Usage

```python
from src.converter.preprocess import preprocess
from src.converter.classifier import classify_lines
from src.converter.characters import extract_characters, CharacterConfig, to_character_entries
from src.converter.emotions import tag_emotions, EmotionConfig
from src.converter.segmenter import segment_events, SegmenterConfig

# Full pipeline
chapters, lines = preprocess("novel.txt")
line_types = classify_lines(lines)
characters, enriched = extract_characters(chapters, lines, line_types, CharacterConfig())
emotions = tag_emotions(lines, line_types, EmotionConfig())
for e, emos in zip(enriched, emotions):
    e.emotions = emos

# Segmentation
seg_cfg = SegmenterConfig(
    threshold=0.5,
    pov_tracking=True,
    min_pov_run_lines=3,
)
events = segment_events(enriched, to_character_entries(characters), seg_cfg)

# events: list[PlotEvent] with Scenario children → ready for VNScript export
for evt in events:
    print(f"{evt.id}: {len(evt.scenarios)} scenarios, chars={evt.characters}")
    for scn in evt.scenarios:
        print(f"  {scn.name} ({scn.time}) - {len(scn.lines)} lines")
```

### Output Structure (matches `schema.py`)

```
PlotEvent (evt_001)
├── chapter: 1
├── characters: ["char_001", "char_002"]
├── scenarios:
│   ├── Scenario (scn_0_000)
│   │   ├── name: "sect courtyard"
│   │   │   ├── time: TimeOfDay.MORNING
│   │   │   ├── background: None
│   │   │   ├── lines: [ScriptLine(...), ...]
│   ├── Scenario (scn_0_001)
│   │   ├── name: "elder's chamber"
│   │   │   ├── time: TimeOfDay.AFTERNOON
```

### Configuration via `configs/pipeline1.yaml`

```yaml
segmenter:
  method: "embedding"            # "texttiling" | "embedding" | "hybrid"
  window_size: 10
  threshold: 0.5
  use_weak_labels: true
  min_segment_length: 3
  time_detection: "rules"        # "rules" | "dateparser"
  pov_tracking: true
  min_pov_run_lines: 5
  min_pov_run_ratio: 0.2
```