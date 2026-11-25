# Repo Structure
```
wn2vn/
├── configs/
│   └── emotions.yaml              # Canonical emotions + Ren'Py expression map
│
├── data/
│   ├── raw/
│   │   ├── wn/                    # Input Web Novel files (.txt, .html, .epub)
│   │   └── vn/                    # Ground-truth Ren'Py .rpy scripts
│   └── output/
│       ├── pipeline1_wn/          # Generated VNScript JSONs (and optional .rpy exports)
│       └── pipeline2_vn/          # Extracted ground-truth VNScript JSONs
│
├── notebooks/
│   ├── prototype.ipynb            # Interactive experimentation & model testing
│   └── evaluation.ipynb           # Score visualizations & error analysis
│
├── src/
│   ├── schema.py                  # Shared Pydantic VNScript contract
│   │
│   ├── converter/                 # PIPELINE 1: Web Novel -> VNScript JSON
│   │   ├── preprocess.py          # Steps 1–3: HTML/text clean, chapter split, Corpus A
│   │   ├── classifier.py          # Step 4: narration / dialogue / monologue
│   │   ├── characters.py          # Steps 5–6: NER, alias merge, coref, speaker attribution
│   │   ├── emotions.py            # Step 7: line-level & trigger-span emotion tagging
│   │   ├── segmenter.py           # Step 8: setting detection & plot event grouping
│   │   ├── exporter.py            # Step 9: JSON serializer & Jinja2 .rpy exporter
│   │   └── pipeline.py            # Orchestrates Steps 1–9 end-to-end
│   │
│   ├── extractor/                 # PIPELINE 2: Ren'Py .rpy -> VNScript JSON
│   │   └── renpy_parser.py        # Parses labels, scenes, sprites/emotions, and lines
│   │
│   └── eval/                      # EVALUATION: Pipeline 1 vs. Pipeline 2
│       ├── aligner.py             # Needleman-Wunsch fuzzy line aligner
│       └── metrics.py             # Weighted scoring (events, speakers, lines, emotions)
│
├── tests/
│   ├── test_preprocess.py         # Unit tests for Steps 1–3
│   ├── test_converter.py          # Unit tests for Steps 4–9
│   ├── test_extractor.py          # Unit tests for Ren'Py parser
│   └── test_eval.py               # Unit tests for alignment & scoring
│
├── main.py                        # CLI entry point to run Pipeline 1, Pipeline 2, or Eval
├── pyproject.toml                 # Pytest & project configuration
├── requirements.txt               # Pinned dependencies (pydantic, trafilatura, spacy, etc.)
└── README.md                      # Setup instructions, dataset notes, and pipeline usage
```

# Setup Guide
1. Create and activate a virtual environment
```
python3 -m venv .venv
```
For Linux/macOS:
```
source .venv/bin/activate
```
For Windows:
```
.venv\Scripts\activate
```

2. Install dependencies (and any NLP models added later)
```
pip install -r requirements.txt
```

3. Verify
```
pytest
```
Will report collected 0 items and exit with status code 5 (no tests collected).