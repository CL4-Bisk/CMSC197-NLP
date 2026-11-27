"""Pipeline 1, Step 5: Character Extraction (NER + Coreference + Alias Merging + Speaker Attribution).

Pipeline:
    1. NER (spaCy en_core_web_trf) → PERSON entities from narration + dialogue
    2. Coreference (fastcoref) → link pronouns to antecedents
    3. Alias merging (rapidfuzz + title stripping) → cluster surface forms per character
    4. Speaker attribution → assign speaker_id to each dialogue line
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from rapidfuzz import fuzz, process

from src.converter.preprocess import Chapter, RawLine
from src.converter.classifier import LineType
from src.schema import CharacterEntry


# --------------------------------------------------------------------------- #
# Default title prefixes (xianxia/xuanhuan + general)
# --------------------------------------------------------------------------- #

DEFAULT_TITLE_PREFIXES: tuple[str, ...] = (
    # Compound prefixes (checked first, longest match wins)
    "young master", "senior brother", "junior brother", "senior sister", "junior sister",
    "sect master", "patriarch", "matriarch", "daoist master",
    # Cultivation titles
    "senior", "elder", "master", "lord", "lady",
    "prince", "princess", "emperor", "empress", "general", "commander",
    "ancestor", "daoist", "venerable", "reverend", "sage", "immortal", "cultivator",
    # Family/relation
    "brother", "sister", "uncle", "aunt", "grandfather", "grandmother",
    "father", "mother", "son", "daughter", "cousin",
    # Generic
    "mr", "mrs", "ms", "dr", "prof", "sir", "madam",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CharacterConfig:
    # NER
    ner_model: Literal["spacy_trf", "spacy_sm", "flair"] = "spacy_sm"

    # Coreference
    coref_model: Literal["fastcoref", "lingmess"] = "fastcoref"

    # Alias merging
    fuzzy_threshold: int = 85
    title_prefixes: tuple[str, ...] = DEFAULT_TITLE_PREFIXES

    # Speaker attribution
    attribution_window: int = 3
    use_alternation: bool = True


# --------------------------------------------------------------------------- #
# Internal data structures
# --------------------------------------------------------------------------- #

@dataclass
class Mention:
    text: str                  # Surface form: "Young Master Lin"
    chapter: int
    paragraph: int
    line_idx: int              # Index in the RawLine list
    char_start: int
    char_end: int
    is_pronoun: bool = False   # True for "he"/"she"/"they"/"him"/"her"

    def source_ref(self):
        from src.schema import SourceRef
        return SourceRef(
            chapter=self.chapter,
            paragraph=self.paragraph,
            char_start=self.char_start,
            char_end=self.char_end,
        )


@dataclass
class Character:
    id: str                    # "char_001"
    name: str                  # Canonical name: "Lin Feng"
    aliases: list[str] = field(default_factory=list)  # All surface forms
    mentions: list[Mention] = field(default_factory=list)

    def add_mention(self, mention: Mention) -> None:
        self.mentions.append(mention)
        if mention.text not in self.aliases:
            self.aliases.append(mention.text)

    def to_entry(self) -> CharacterEntry:
        return CharacterEntry(
            id=self.id,
            name=self.name,
            aliases=self.aliases,
        )


# Wrapper for enriched lines (SRP: doesn't modify RawLine)
@dataclass
class CharacterEnrichedLine:
    raw: RawLine
    line_type: LineType
    speaker_id: Optional[str] = None

    # Delegate RawLine properties
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


# --------------------------------------------------------------------------- #
# NER (spaCy)
# --------------------------------------------------------------------------- #

_PERSON_LABELS = {"PERSON", "PER"}

def _load_spacy_ner(model: Literal["en_core_web_trf", "en_core_web_sm"] = "en_core_web_sm"):
    import spacy
    try:
        return spacy.load(model)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model}' not found. Run: python -m spacy download {model}"
        )


def extract_person_entities(
    chapters: list[Chapter],
    lines: list[RawLine],
    nlp,
) -> list[Mention]:
    """Run NER on all text, return Mention objects for PERSON entities."""
    mentions: list[Mention] = []

    # Build text with line mapping
    # We process chapter by chapter to keep context
    for ch in chapters:
        # Combine paragraph texts for NER context
        full_text = "\n\n".join(p.text for p in ch.paragraphs)
        doc = nlp(full_text)

        # Map entity spans back to lines
        # Strategy: for each PERSON ent, find which line(s) it overlaps
        line_idx = 0
        for line in lines:
            if line.chapter != ch.index:
                continue

        # Simpler: process each paragraph separately, track line indices
        line_ptr = 0
        for para in ch.paragraphs:
            para_doc = nlp(para.text)
            for ent in para_doc.ents:
                if ent.label_ not in _PERSON_LABELS:
                    continue
                # Find the line this entity falls in
                while line_ptr < len(lines):
                    ln = lines[line_ptr]
                    if ln.chapter != ch.index or ln.paragraph != para.index:
                        line_ptr += 1
                        continue
                    # Check overlap: entity in para.text overlaps line's char span
                    if ent.start_char < ln.char_end and ent.end_char > ln.char_start:
                        mentions.append(Mention(
                            text=ent.text,
                            chapter=ch.index,
                            paragraph=para.index,
                            line_idx=line_ptr,
                            char_start=ent.start_char,
                            char_end=ent.end_char,
                            is_pronoun=False,
                        ))
                        break
                    line_ptr += 1

    return mentions


# --------------------------------------------------------------------------- #
# Coreference (fastcoref)
# --------------------------------------------------------------------------- #

def _load_fastcoref(model_name: str = "fastcoref"):
    try:
        import fastcoref
        return fastcoref.LingMessCoref() if model_name == "lingmess" else fastcoref.FCoref()
    except ImportError:
        raise RuntimeError("fastcoref not installed. Run: pip install fastcoref")
    except Exception as e:
        # Version conflict or model loading issue - return None to skip coref
        import warnings
        warnings.warn(f"fastcoref unavailable ({e}), skipping coreference resolution")
        return None


def resolve_coreferences(
    chapters: list[Chapter],
    mentions: list[Mention],
    coref_model,
) -> list[Mention]:
    """Add pronoun mentions resolved via coreference."""
    # Build full text per chapter for coref
    chapter_texts = {}
    for ch in chapters:
        chapter_texts[ch.index] = "\n\n".join(p.text for p in ch.paragraphs)

    new_mentions = []
    for ch_idx, text in chapter_texts.items():
        if not text.strip():
            continue
        try:
            preds = coref_model.predict(texts=[text])
            clusters = preds[0].get("clusters", [])
        except Exception:
            continue

        # For each cluster, find the representative (longest non-pronoun mention)
        # and assign its name to pronoun mentions
        for cluster in clusters:
            # cluster = list of (start_char, end_char) in text
            cluster_mentions = []
            for start, end in cluster:
                span_text = text[start:end]
                cluster_mentions.append((start, end, span_text))

            # Find best antecedent (longest non-pronoun)
            antecedent = None
            for s, e, t in cluster_mentions:
                if not _is_pronoun(t) and (antecedent is None or len(t) > len(antecedent[2])):
                    antecedent = (s, e, t)

            if antecedent is None:
                continue

            ant_name = antecedent[2]
            # Add pronoun mentions linked to antecedent
            for s, e, t in cluster_mentions:
                if _is_pronoun(t):
                    # Find which line this falls in
                    line_idx = _find_line_for_offset(ch_idx, s, e, chapters, mentions)
                    if line_idx is not None:
                        new_mentions.append(Mention(
                            text=t,
                            chapter=ch_idx,
                            paragraph=0,  # will be corrected below
                            line_idx=line_idx,
                            char_start=s,
                            char_end=e,
                            is_pronoun=True,
                        ))
    return new_mentions


def _is_pronoun(text: str) -> bool:
    return text.lower() in {
        "he", "she", "they", "him", "her", "them", "his", "hers", "theirs",
        "himself", "herself", "themselves", "it", "its", "itself",
    }


def _find_line_for_offset(ch_idx: int, start: int, end: int,
                          chapters: list[Chapter], mentions: list[Mention]) -> Optional[int]:
    # Approximate: find line in same chapter whose char span is closest
    best_idx = None
    best_dist = float('inf')
    for i, ln in enumerate(mentions):
        if ln.chapter != ch_idx:
            continue
        dist = min(abs(ln.char_start - start), abs(ln.char_end - end))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


# --------------------------------------------------------------------------- #
# Alias Merging (rapidfuzz + title stripping)
# --------------------------------------------------------------------------- #

def _strip_titles(name: str, title_prefixes: tuple[str, ...]) -> str:
    """Remove title prefixes: 'Young Master Lin' → 'Lin'.

    Prefixes are checked in order of descending length to match compounds first.
    """
    lower = name.lower().strip()
    # Sort by length descending to match longest prefixes first
    for prefix in sorted(title_prefixes, key=len, reverse=True):
        if lower.startswith(prefix + " "):
            return name[len(prefix) + 1:].strip()
        if lower == prefix:
            return ""
    return name


def _normalize_for_matching(name: str, title_prefixes: tuple[str, ...]) -> str:
    """Key for fuzzy matching: strip titles, lowercase, remove punctuation."""
    stripped = _strip_titles(name, title_prefixes)
    return re.sub(r"[^\w\s]", "", stripped).lower().strip()


def _last_name(name: str) -> str:
    """Extract surname/family name.

    For Chinese names (2-3 parts, first part 1-4 chars): first part is surname.
    For Western names: last part is surname.
    """
    parts = name.split()
    if not parts:
        return ""
    # Heuristic: Chinese names have 2-3 parts, first part 1-4 chars (surname in pinyin)
    if len(parts) >= 2 and len(parts[0]) <= 4:
        return parts[0].lower()  # Chinese: surname first
    return parts[-1].lower()  # Western: surname last


def merge_aliases(
    characters: list[Character],
    threshold: int = 85,
    title_prefixes: tuple[str, ...] = DEFAULT_TITLE_PREFIXES,
) -> list[Character]:
    """Cluster characters by fuzzy name matching after title stripping."""
    if not characters:
        return characters

    # Build normalized keys (full stripped name) and last-name keys
    full_keys = [_normalize_for_matching(c.name, title_prefixes) for c in characters]
    last_keys = [_last_name(_strip_titles(c.name, title_prefixes)) for c in characters]

    used = [False] * len(characters)
    merged: list[Character] = []

    for i, char in enumerate(characters):
        if used[i]:
            continue
        used[i] = True
        cluster = [char]
        key_i_full = full_keys[i]
        key_i_last = last_keys[i]

        for j in range(i + 1, len(characters)):
            if used[j]:
                continue
            # Match on full normalized name OR last name
            score_full = fuzz.ratio(key_i_full, full_keys[j])
            score_last = fuzz.ratio(key_i_last, last_keys[j]) if key_i_last and last_keys[j] else 0
            if score_full >= threshold or score_last >= threshold:
                used[j] = True
                cluster.append(characters[j])

        # Merge cluster: pick longest name as canonical
        canonical = max(cluster, key=lambda c: len(c.name))
        for other in cluster:
            if other is not canonical:
                canonical.aliases.extend(other.aliases)
                canonical.mentions.extend(other.mentions)
        # Deduplicate aliases
        canonical.aliases = list(dict.fromkeys(canonical.aliases))
        merged.append(canonical)

    return merged


# --------------------------------------------------------------------------- #
# Speaker Attribution
# --------------------------------------------------------------------------- #

_SAID_VERBS = {
    "said", "says", "asked", "asks", "replied", "replies", "answered", "answers",
    "whispered", "whispers", "shouted", "shouts", "yelled", "yells",
    "murmured", "murmurs", "muttered", "mutters", "called", "calls",
    "cried", "cries", "screamed", "screams", "stated", "states",
    "declared", "declares", "announced", "announces", "mentioned", "mentions",
    "noted", "notes", "observed", "observes", "remarked", "remarks",
}

def _extract_speaker_from_tag(text: str, characters: list[Character]) -> Optional[str]:
    """Parse 'Lin Feng said' / 'said Lin Feng' patterns."""
    # Pattern: Name + said_verb
    for verb in _SAID_VERBS:
        # "Lin Feng said" - capture everything before the verb
        pattern1 = rf"^(.+?)\s+{verb}\b"
        # "said Lin Feng" - capture everything after the verb (rest of line)
        pattern2 = rf"^{verb}\s+(.+)$"
        for pattern in (pattern1, pattern2):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Fuzzy match to known character
                best = _match_candidate_to_character(candidate, characters)
                if best:
                    return best.id
    return None


def _match_candidate_to_character(candidate: str, characters: list[Character]) -> Optional[Character]:
    """Fuzzy match a candidate name to existing characters (with title stripping)."""
    cand_norm = _normalize_for_matching(candidate, DEFAULT_TITLE_PREFIXES)
    cand_last = _last_name(cand_norm)
    best_char = None
    best_score = 0
    for char in characters:
        for alias in char.aliases:
            alias_norm = _normalize_for_matching(alias, DEFAULT_TITLE_PREFIXES)
            alias_last = _last_name(alias_norm)
            # Full name match
            score = fuzz.ratio(cand_norm, alias_norm)
            # Last name match (for "Young Master Lin" vs "Lin Feng")
            if cand_last and alias_last:
                score = max(score, fuzz.ratio(cand_last, alias_last) + 10)  # slight boost
            if score > best_score and score >= 80:
                best_score = score
                best_char = char
    return best_char


def _find_nearest_character(line_idx: int, lines: list[CharacterEnrichedLine],
                           characters: list[Character], window: int) -> Optional[Character]:
    """Search backward for nearest named character in narration."""
    for i in range(line_idx - 1, max(-1, line_idx - window - 1), -1):
        ln = lines[i]
        if ln.line_type == LineType.NARRATION and not ln.quoted:
            # Check if this line contains a character name
            for char in characters:
                for alias in char.aliases:
                    if alias.lower() in ln.text.lower():
                        return char
    return None


def attribute_speakers(
    lines: list[CharacterEnrichedLine],
    characters: list[Character],
    config: CharacterConfig,
) -> None:
    """Assign speaker_id to dialogue lines (in-place mutation of speaker_id)."""
    last_speakers: list[str] = []  # For alternation heuristic

    for i, ln in enumerate(lines):
        if ln.line_type != LineType.DIALOGUE or ln.quoted is False:
            continue

        speaker_id = None

        # 1. Explicit tag on previous narration line
        if i > 0:
            prev = lines[i - 1]
            if prev.line_type == LineType.NARRATION:
                speaker_id = _extract_speaker_from_tag(prev.text, characters)

        # 2. Coref: if current line starts with pronoun, use last speaker
        if speaker_id is None and ln.text and _is_pronoun(ln.text.split()[0].lower()):
            if last_speakers:
                speaker_id = last_speakers[-1]

        # 3. Nearest named character in window
        if speaker_id is None:
            char = _find_nearest_character(i, lines, characters, config.attribution_window)
            if char:
                speaker_id = char.id

        # 4. Alternation heuristic (A/B/A/B for two-speaker dialogue)
        if speaker_id is None and config.use_alternation and len(last_speakers) >= 2:
            # If only 2 characters in recent dialogue, alternate
            recent_chars = set(last_speakers[-4:])
            if len(recent_chars) == 2:
                # Pick the one who didn't speak last
                for cid in recent_chars:
                    if cid != last_speakers[-1]:
                        speaker_id = cid
                        break

        if speaker_id:
            ln.speaker_id = speaker_id
            last_speakers.append(speaker_id)
            # Keep only recent history
            if len(last_speakers) > 10:
                last_speakers.pop(0)


# --------------------------------------------------------------------------- #
# Main Pipeline Function
# --------------------------------------------------------------------------- #

def extract_characters(
    chapters: list[Chapter],
    lines: list[RawLine],
    line_types: list[LineType],
    config: CharacterConfig | None = None,
) -> tuple[list[Character], list[CharacterEnrichedLine]]:
    """
    Full character extraction pipeline.

    Args:
        chapters: From preprocess.clean_and_split_chapters()
        lines: From preprocess.split_lines()
        line_types: From classifier.classify_lines()
        config: CharacterConfig (optional)

    Returns:
        (characters, enriched_lines)
        - characters: list[Character] with id, name, aliases, mentions
        - enriched_lines: list[CharacterEnrichedLine] with speaker_id for dialogue
    """
    config = config or CharacterConfig()

    # 0. Wrap lines with line_type
    enriched = [
        CharacterEnrichedLine(raw=ln, line_type=lt)
        for ln, lt in zip(lines, line_types)
    ]

    # 1. NER
    nlp = _load_spacy_ner("en_core_web_trf" if config.ner_model == "spacy_trf" else "en_core_web_sm")
    mentions = extract_person_entities(chapters, lines, nlp)

    # 2. Coreference (optional - skip if unavailable)
    coref = _load_fastcoref("lingmess" if config.coref_model == "lingmess" else "fastcoref")
    if coref is not None:
        coref_mentions = resolve_coreferences(chapters, mentions, coref)
        mentions.extend(coref_mentions)

    # 3. Build initial characters from mentions
    char_map: dict[str, Character] = {}  # normalized name -> Character
    char_counter = 0

    for mention in mentions:
        key = _normalize_for_matching(mention.text, config.title_prefixes)
        if not key:
            continue
        if key not in char_map:
            char_counter += 1
            char_map[key] = Character(id=f"char_{char_counter:03d}", name=mention.text)
        char_map[key].add_mention(mention)

    characters = list(char_map.values())

    # 4. Alias merging
    characters = merge_aliases(characters, config.fuzzy_threshold, config.title_prefixes)

    # Re-assign sequential IDs after merge
    for idx, char in enumerate(characters):
        char.id = f"char_{idx + 1:03d}"

    # 5. Speaker attribution
    attribute_speakers(enriched, characters, config)

    return characters, enriched


def to_character_entries(characters: list[Character]) -> list[CharacterEntry]:
    """Convert internal Character objects to schema.CharacterEntry for export."""
    return [c.to_entry() for c in characters]


if __name__ == "__main__":
    # Quick test
    from src.converter.preprocess import preprocess
    from src.converter.classifier import classify_lines

    text = (
        "Chapter 1\n\n"
        "Lin Feng stared at the blade. “I won't lose,” he said.\n\n"
        "*Why is my heart racing?*\n\n"
        "The elder nodded. \"Good.\"\n\n"
        "Young Master Lin smiled."
    )
    chapters, lines = preprocess(text, is_text=True)
    line_types = classify_lines(lines)
    characters, enriched = extract_characters(chapters, lines, line_types)

    print("Characters:")
    for c in characters:
        print(f"  {c.id}: {c.name} | aliases={c.aliases}")
    print("\nEnriched lines:")
    for e in enriched:
        spk = e.speaker_id or "-"
        print(f"  [{e.line_type.value}] spk={spk} | {e.text}")