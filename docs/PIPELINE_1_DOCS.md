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