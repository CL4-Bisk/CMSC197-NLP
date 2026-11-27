# Web Novel to Visual Novel Script (wn2vn)

A configurable NLP pipeline that converts web novels into structured visual novel scripts (VNScript JSON), with a second pipeline for extracting ground-truth scripts from Ren'Py (.rpy) files, and an evaluation framework for comparing them.

## Pipeline Overview

```
Pipeline 1 (Converter): Web Novel → VNScript JSON
├── Step 1: Load raw text (.txt, .html, .epub, or folder of chapters)
├── Step 2: Clean noise, detect scene breaks, split into chapters
├── Step 3: Corpus A line split (paragraph + quote boundaries)
├── Step 4: Classify lines → narration / dialogue / monologue
├── Step 5: Extract characters (NER + coref + alias merge + speaker attribution)
├── Step 6: Tag emotions (NRC lexicon / DistilRoBERTa / GoEmotions)
├── Step 7: Segment into scenarios (setting/time) + plot events
└── Step 8: Export to VNScript JSON + optional Ren'Py .rpy

Pipeline 2 (Extractor): Ren'Py .rpy → VNScript JSON
└── Parse labels, scenes, sprites, expressions, and dialogue lines

Evaluation: Align Pipeline 1 output with Pipeline 2 ground truth
└── Needleman-Wunsch fuzzy alignment + weighted scoring
```

## Quick Start

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install spaCy Models

```bash
# Small model (fast, lower NER accuracy) - DEFAULT
python -m spacy download en_core_web_sm

# Transformer model (slower, higher NER accuracy) - OPTIONAL
python -m spacy download en_core_web_trf
```

### 4. Run Pipeline 1

```bash
# Process a single file
python main.py path/to/novel.txt

# Process a folder (one file per chapter)
python main.py path/to/novel_folder/

# Specify output file
python main.py path/to/novel.txt -o output/script.json

# Use custom config
python main.py --config configs/custom.yaml path/to/novel.txt

# Process raw text directly
python main.py "Chapter 1\n\nHe said \"Hello.\"" --is-text
```

### 5. Run Pipeline 2 (VN Extractor)

Pipeline 2 extracts ground-truth VNScript JSON from Ren'Py (.rpy) script files for evaluation comparison.

```bash
# Extract from a single .rpy file
python main.py --pipeline 2 path/to/script.rpy

# Extract from a folder of .rpy files (one file per label/event)
python main.py --pipeline 2 path/to/vn_scripts/

# Specify output file
python main.py --pipeline 2 path/to/script.rpy -o output/ground_truth.json
```

**What Pipeline 2 extracts:**

| Ren'Py Element | VNScript Mapping |
|----------------|------------------|
| `label` statements | `PlotEvent.label` (event identifier) |
| `scene` / `show` with background | `Scenario.background` |
| `show` / `hide` character sprites | `CharacterEntry.image_tag`, `renpy_var` |
| Character definitions (`define e = Character(...)`) | `CharacterEntry.name`, `renpy_var`, `image_tag` |
| Dialogue lines (`e "Hello"`) | `ScriptLine` with `speaker`, `type=dialogue` |
| Narration (`"Text"` or `narrator "Text"`) | `ScriptLine` with `type=narration` |
| Monologue / internal thought (`"Text"` with specific character) | `ScriptLine` with `type=monologue` |
| Sprite expressions (`show eileen happy`) | `ScriptLine.emotions` (mapped via `configs/emotions.yaml`) |
| `scene` transitions | Scenario boundaries (setting changes) |
| `call` / `jump` to labels | Event flow (optional) |

**Output:** Same VNScript JSON schema as Pipeline 1, with `source_type: "visual_novel"` and `pipeline: "extractor"`.

### 6. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_preprocess.py -v

# Run slow integration tests (require spaCy models)
pytest -m slow -v
```

## Project Structure

```
wn2vn/
├── configs/
│   ├── pipeline1.yaml          # Default Pipeline 1 configuration
│   └── emotions.yaml           # Canonical emotions + Ren'Py expression map
│
├── data/
│   ├── raw/
│   │   ├── wn/                 # Input Web Novel files (.txt, .html, .epub)
│   │   └── vn/                 # Ground-truth Ren'Py .rpy scripts
│   └── output/
│       ├── pipeline1_wn/       # Generated VNScript JSONs (and optional .rpy exports)
│       └── pipeline2_vn/       # Extracted ground-truth VNScript JSONs
│
├── notebooks/
│   ├── prototype.ipynb         # Interactive experimentation & model testing
│   └── evaluation.ipynb        # Score visualizations & error analysis
│
├── src/
│   ├── schema.py               # Shared Pydantic VNScript contract
│   │
│   ├── converter/              # PIPELINE 1: Web Novel -> VNScript JSON
│   │   ├── preprocess.py       # Steps 1-3: HTML/text clean, chapter split, Corpus A
│   │   ├── classifier.py       # Step 4: narration / dialogue / monologue
│   │   ├── characters.py       # Steps 5-6: NER, alias merge, coref, speaker attribution
│   │   ├── emotions.py         # Step 7: line-level & trigger-span emotion tagging
│   │   ├── segmenter.py        # Step 8: setting detection & plot event grouping
│   │   ├── exporter.py         # Step 9: JSON serializer & Jinja2 .rpy exporter
│   │   └── pipeline.py         # Orchestrates Steps 1-9 end-to-end
│   │
│   ├── extractor/              # PIPELINE 2: Ren'Py .rpy -> VNScript JSON
│   │   └── renpy_parser.py     # Parses labels, scenes, sprites/emotions, and lines
│   │
│   └── eval/                   # EVALUATION: Pipeline 1 vs. Pipeline 2
│       ├── aligner.py          # Needleman-Wunsch fuzzy line aligner
│       └── metrics.py          # Weighted scoring (events, speakers, lines, emotions)
│
├── tests/
│   ├── test_preprocess.py      # Unit tests for Steps 1-3
│   ├── test_classifier.py      # Unit tests for Step 4
│   ├── test_characters.py      # Unit tests for Steps 5-6
│   ├── test_emotions.py        # Unit tests for Step 7
│   ├── test_segmenter.py       # Unit tests for Step 8
│   ├── test_extractor.py       # Unit tests for Ren'Py parser
│   └── test_eval.py            # Unit tests for alignment & scoring
│
├── main.py                     # CLI entry point for Pipeline 1, Pipeline 2, or Eval
├── pyproject.toml              # Pytest & project configuration
├── requirements.txt            # Pinned dependencies
└── README.md                   # This file
```

## Configuration

Pipeline 1 is configured via YAML (`configs/pipeline1.yaml` by default):

```yaml
preprocess:
  single_newline_paragraphs: false
  drop_noise: true
  extra_noise_patterns: []

classifier:
  model: "rules"              # "rules" or "bart_zeroshot"
  confidence_threshold: 0.7
  device: "auto"              # "cpu", "cuda", "mps", "auto"
  batch_size: 16

characters:
  ner_model: "spacy_sm"       # "spacy_sm", "spacy_trf", "flair"
  coref_model: "fastcoref"    # "fastcoref", "lingmess"
  fuzzy_threshold: 85
  title_prefixes: []          # custom title prefixes
  attribution_window: 3
  use_alternation: true

emotions:
  model: "nrc"                # "nrc", "distilroberta", "go_emotions"
  return_spans: true
  min_words_for_emotion: 1
  dominant_top_k: 1
  device: "auto"
  batch_size: 16

segmenter:
  method: "embedding"         # "embedding", "texttiling", "hybrid"
  window_size: 10
  threshold: 0.5
  use_weak_labels: true
  min_segment_length: 3
  time_detection: "rules"     # "rules", "dateparser"
  setting_prototypes: "auto"
  pov_tracking: true
  min_pov_run_lines: 5
  min_pov_run_ratio: 0.2
```

### Model Options

| Component | Options | Notes |
|-----------|---------|-------|
| **Classifier** | `rules` (default), `bart_zeroshot` | BART requires `transformers` + `torch` |
| **NER** | `spacy_sm` (default), `spacy_trf`, `flair` | `spacy_trf` needs `en_core_web_trf` model |
| **Coreference** | `fastcoref` (default), `lingmess` | Both require `fastcoref` package |
| **Emotions** | `nrc` (default), `distilroberta`, `go_emotions` | ML models need `transformers` + `torch` |
| **Segmentation** | `embedding` (default), `texttiling`, `hybrid` | Embedding uses `sentence-transformers` |

## Output Format (VNScript JSON)

```json
{
  "meta": {
    "title": "Novel Title",
    "source_type": "web_novel",
    "pipeline": "converter",
    "schema_version": "1.0.0",
    "generated_at": "2024-01-15T10:30:00Z",
    "models": {
      "preprocess": "trafilatura+regex",
      "classifier": "rules",
      "characters": "spacy_sm+rapidfuzz",
      "emotions": "nrc",
      "segmenter": "embedding"
    }
  },
  "characters": [
    {
      "id": "char_001",
      "name": "Lin Feng",
      "aliases": ["Lin Feng", "Young Master Lin", "Feng-er"],
      "renpy_var": null,
      "image_tag": null
    }
  ],
  "events": [
    {
      "id": "evt_001",
      "chapter": 1,
      "summary": null,
      "label": null,
      "characters": ["char_001", "char_002"],
      "scenarios": [
        {
          "id": "scn_001_000",
          "name": "sect courtyard",
          "time": "morning",
          "background": null,
          "lines": [
            {
              "id": "ln_001_001_000",
              "text": "Lin Feng drew his sword.",
              "type": "narration",
              "speaker": null,
              "emotions": [],
              "source_ref": {"chapter": 1, "paragraph": 1, "char_start": 0, "char_end": 20}
            },
            {
              "id": "ln_001_001_001",
              "text": "I will win!",
              "type": "dialogue",
              "speaker": "char_001",
              "emotions": [
                {"label": "joy", "score": 0.8, "trigger": {"text": "win", "start": 8, "end": 11}}
              ],
              "source_ref": {"chapter": 1, "paragraph": 1, "char_start": 22, "char_end": 31}
            }
          ]
        }
      ]
    }
  ]
}
```

## Evaluation Framework

Compare Pipeline 1 output (predicted) against Pipeline 2 output (ground truth).

```bash
# Run evaluation
python main.py --eval path/to/pipeline1_output.json path/to/pipeline2_output.json

# With custom weights
python main.py --eval pred.json truth.json --weights events=0.3,speakers=0.3,lines=0.2,emotions=0.2
```

**Alignment:** Needleman-Wunsch algorithm with fuzzy line matching (character + text similarity).

**Metrics (weighted):**
- **Events** (0.3): Event boundary accuracy, scenario grouping
- **Speakers** (0.3): Character attribution precision/recall
- **Lines** (0.2): Line-type classification (narration/dialogue/monologue)
- **Emotions** (0.2): Emotion label + trigger span overlap

Output: JSON report with per-event scores, confusion matrices, and aggregate F1.

## Emotion System

Canonical emotions (from `configs/emotions.yaml`):

| Emotion | Ren'Py Expression | NRC Source |
|---------|-------------------|------------|
| `neutral` | `normal` | - |
| `joy` | `smile` | joy, positive, trust |
| `sadness` | `sad` | sadness, negative |
| `anger` | `angry` | anger, disgust |
| `fear` | `scared` | fear |
| `surprise` | `surprised` | surprise, anticipation |
| `disgust` | `disgusted` | disgust |

## Input Formats

| Format | Description |
|--------|-------------|
| `.txt` / `.md` | Plain text, one chapter per file (folder) or single file |
| `.html` / `.htm` / `.xhtml` | Web pages (uses trafilatura for main content extraction) |
| `.epub` | E-book format (reads spine order) |
| Folder | Directory of chapter files (sorted naturally: ch_1, ch_2, ch_10) |

## Development

### Adding a New Test

```bash
# Create test file in tests/
# Run with: pytest tests/test_new_feature.py -v
```

### Code Style

```bash
# Lint (optional)
ruff check src/

# Type check (optional)
mypy src/
```

### Optional Dependencies

For advanced ML models, install the optional dependencies:

```bash
# For BART zero-shot classifier, DistilRoBERTa, GoEmotions
pip install transformers torch accelerate

# For Flair NER (alternative to spaCy)
pip install flair
```

## Troubleshooting

### spaCy model not found
```bash
python -m spacy download en_core_web_sm
# or for transformer model:
python -m spacy download en_core_web_trf
```

### fastcoref unavailable
```bash
pip install fastcoref
# Note: may require specific torch version compatibility
```

### Import errors for sentence-transformers
```bash
pip install sentence-transformers
```

### Memory issues with transformer models
- Use `device: "cpu"` in config
- Reduce `batch_size` in classifier/emotions/segmenter configs
- Use smaller models (e.g., `spacy_sm` instead of `spacy_trf`)

## License

MIT License - see LICENSE file for details.