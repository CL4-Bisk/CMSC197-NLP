"""Pipeline 1, Steps 1-3: extract raw text, clean + chapter-split, Corpus A line split.

    load_raw(path)                 -> str                (Step 1, one file)
    load_book(folder)              -> str                (Step 1, folder = one novel)
    clean_and_split_chapters(text) -> list[Chapter]      (Step 2)
    split_lines(chapters)          -> list[RawLine]      (Step 3)
    preprocess(path_or_text)       -> (chapters, lines)  all three

Conventions
-----------
* Chapter and paragraph indices are 1-based.
* Italics are kept in `Paragraph.text` as `*...*` (trafilatura's markdown output
  uses the same marker) so Step 4 can use them for monologue detection.
  `RawLine.text` is marker-free; `RawLine.italic` says whether the whole line
  was italic.
* `RawLine.char_start/char_end` are offsets into `Paragraph.text` (markers
  included), pointing at the line's content without its quote marks.
* Scene breaks (`***`, `---`, `* * *`, ...) are not kept as text; they set
  `Paragraph.scene_break_before` (and `RawLine.scene_break_before` on the first
  line of that paragraph). Step 8 can use them as weak boundary labels.
"""
from __future__ import annotations

import html as htmllib
import posixpath
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from src.schema import SourceRef

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

# opener -> closer. Straight quotes open and close with the same character.
DEFAULT_QUOTES: dict[str, str] = {
    "\u201c": "\u201d",  # “ ”
    '"': '"',
    "\u300c": "\u300d",  # 「 」
    "\u300e": "\u300f",  # 『 』
    "\u00ab": "\u00bb",  # « »
}

@dataclass(frozen=True)
class PreprocessConfig:
    # True when the source puts one paragraph per line (many .txt web novels).
    # False follows Corpus A literally: paragraphs are separated by blank lines.
    single_newline_paragraphs: bool = False
    drop_noise: bool = True
    extra_noise_patterns: tuple[str, ...] = ()
    quote_pairs: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_QUOTES))

@dataclass
class Paragraph:
    index: int
    text: str
    scene_break_before: bool = False

@dataclass
class Chapter:
    index: int
    title: str | None = None
    paragraphs: list[Paragraph] = field(default_factory=list)

@dataclass
class RawLine:
    chapter: int
    paragraph: int
    text: str                 # marker-free, no surrounding quote marks
    quoted: bool              # True if it was inside quote marks
    italic: bool              # True if every character of it was italic
    char_start: int
    char_end: int
    scene_break_before: bool = False

    def source_ref(self) -> SourceRef:
        return SourceRef(
            chapter=self.chapter,
            paragraph=self.paragraph,
            char_start=self.char_start,
            char_end=self.char_end,
        )

# --------------------------------------------------------------------------- #
# Step 1: extract raw text
# --------------------------------------------------------------------------- #

_DOC_BOUNDARY = "\f"  # inserted between EPUB documents

def _read_text_file(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes  # installed with trafilatura

        best = from_bytes(data).best()
        return str(best) if best is not None else data.decode("latin-1")

def _html_fallback(raw_html: str) -> str:
    """Crude tag stripper used only when trafilatura returns nothing."""
    s = re.sub(r"(?is)<(script|style|nav|footer|header|noscript)\b.*?</\1>", " ", raw_html)
    s = re.sub(r"(?is)<(i|em)\b[^>]*>(.*?)</\1>", r"*\2*", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</(p|div|h[1-6]|li|blockquote)>", "\n\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return htmllib.unescape(s)

def html_to_text(raw_html: str) -> str:
    """Main-text extraction. Keeps italics as *...* and headings as '# ...'."""
    try:
        import trafilatura

        out = trafilatura.extract(
            raw_html,
            output_format="markdown",
            include_formatting=True,
            include_comments=False,
            include_tables=False,
            include_images=False,
            include_links=False,
            favor_recall=True,
        )
    except ImportError:
        out = None
    return out if out and out.strip() else _html_fallback(raw_html)

def _epub_to_text(path: Path) -> str:
    docs: list[str] = []
    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile").get("full-path")
        opf = ET.fromstring(z.read(rootfile))
        base = posixpath.dirname(rootfile)
        manifest = {i.get("id"): i for i in opf.iterfind(".//{*}manifest/{*}item")}
        for ref in opf.iterfind(".//{*}spine/{*}itemref"):
            item = manifest.get(ref.get("idref"))
            if item is None or "html" not in (item.get("media-type") or ""):
                continue
            name = posixpath.normpath(posixpath.join(base, unquote(item.get("href"))))
            try:
                raw = z.read(name).decode("utf-8", errors="replace")
            except KeyError:
                continue
            text = html_to_text(raw)
            if text.strip():
                docs.append(text)
    return f"\n\n{_DOC_BOUNDARY}\n\n".join(docs)

def load_raw(path: str | Path) -> str:
    """Step 1. Read .txt/.md, .html/.htm or .epub into one raw string."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm", ".xhtml"}:
        return html_to_text(_read_text_file(path))
    if suffix == ".epub":
        return _epub_to_text(path)
    return _read_text_file(path)

_BOOK_SUFFIXES = {".txt", ".md", ".html", ".htm", ".xhtml", ".epub"}
_IGNORED_STEMS = {"readme", "license", "licence"}

def _natural_key(name: str) -> list:
    """'chapter_2' sorts before 'chapter_10'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name.lower())]

def book_files(novel_dir: str | Path) -> list[Path]:
    """Readable chapter files directly inside `novel_dir`, in reading order.

    Everything else is ignored: subfolders, hidden files, images, README/LICENSE,
    and any other file type.
    """
    novel_dir = Path(novel_dir)
    files = [
        f
        for f in novel_dir.iterdir()
        if f.is_file()
        and not f.name.startswith(".")
        and f.suffix.lower() in _BOOK_SUFFIXES
        and f.stem.lower() not in _IGNORED_STEMS
    ]
    return sorted(files, key=lambda f: _natural_key(f.name))

def load_book(novel_dir: str | Path) -> str:
    """Step 1 for a novel stored one-file-per-chapter in a folder.

    Files are read in natural filename order and joined with a document boundary,
    so each file starts a new chapter even when it has no heading of its own.
    """
    files = book_files(novel_dir)
    if not files:
        raise ValueError(f"No readable chapter files in {novel_dir}")
    texts = [t for t in (load_raw(f) for f in files) if t.strip()]
    return f"\n\n{_DOC_BOUNDARY}\n\n".join(texts)

def load_source(path: str | Path) -> str:
    """A folder is one novel (load_book); a file is read with load_raw."""
    path = Path(path)
    return load_book(path) if path.is_dir() else load_raw(path)

# --------------------------------------------------------------------------- #
# Step 2: clean + chapter split
# --------------------------------------------------------------------------- #

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)

_SCENE_BREAK_RE = re.compile(r"^\s*(?:[*\uff0a#~=\-\u2013\u2014\u2022\u25c6\u25c7_+.]\s*){3,}$")

# Whole-paragraph noise (author/translator notes, site boilerplate).
_NOISE_PATTERNS = [
    # Long forms of notes
    r"^[\[(]?\s*(?:translator'?s?|editor'?s?|author'?s?|proofreader'?s?)\s+notes?\b",
    r"^[\[(]?\s*note from (?:the )?(?:author|translator|editor)\b",
    # Short forms need a colon/bracket so "An old man..." is safe
    r"^[\[(]?\s*(?:a/n|t/n|tl/n|tln|tn|tl|an|pr|ed)\s*[:\uff1a\])]",
    r"^(?:translator|editor|proofread(?:er)?|source|original author)\s*[:\uff1a]",
    r"^(?:translated|edited|proofread|proofreaded) by\b",
    r"^(?:previous|prev|next)\s+chapter\b",
    r"^\W*(?:prev(?:ious)?|next|index|toc|table of contents)\W*(?:\|\W*(?:prev(?:ious)?|next|index|toc|table of contents)\W*)*$",
    r"^(?:copyright\b|\u00a9)",
]
# Promo lines are only dropped when short, so prose that mentions them survives.
_PROMO_RE = re.compile(
    r"patreon|ko-?fi|discord|buy me a coffee|novelupdates|wuxiaworld|webnovel|royal ?road"
    r"|read (?:the )?(?:latest|more|advance)|support (?:the )?(?:author|translator)"
    r"|join our|please (?:vote|rate|leave a review|support)",
    re.I,
)
_PROMO_MAX_LEN = 200

# Inline notes: [TN: ...]  (A/N: ...)
_INLINE_NOTE_RE = re.compile(r"[\[(]\s*(?:t/?n|a/?n|tl note|ed note)\s*[:\uff1a][^\])]*[\])]", re.I)

_NUM_WORDS = (
    r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)[\s-]?)?"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
    r"|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
)
_CHAPTER_RE = re.compile(
    rf"^(?:chapter|chap\.?|ch\.?|episode|ep\.?)\s*(?:\d+(?:\.\d+)?|[ivxlcdm]+|{_NUM_WORDS})\b"
    r"\s*(?:[:.\-\u2013\u2014)]\s*)?.*$",
    re.I,
)
_SPECIAL_HEADING_RE = re.compile(
    r"^(?:prologue|epilogue|interlude|prelude|afterword|foreword|extra|side story|omake)\b"
    r"\s*(?:[:.\-\u2013\u2014]\s*.*)?$",
    re.I,
)
_HEADING_MAX_LEN = 120

def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r"[ ]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _strip_markdown(block: str) -> str:
    """Drop markdown structure but keep single-asterisk italics."""
    s = re.sub(r"(?m)^\s*(?:#{1,6}|>)\s*", "", block)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s, flags=re.S)  # bold
    s = re.sub(r"__(.+?)__", r"\1", s, flags=re.S)       # underline
    s = re.sub(r"(?m)^\s*[-+]\s+", "", s)                # list bullets
    return s

def _collapse(block: str) -> str:
    return re.sub(r"\s+", " ", block).strip()

def _is_noise(text: str, patterns: list[re.Pattern]) -> bool:
    plain = text.replace("*", "").strip()
    if not plain:
        return True
    if any(p.search(plain) for p in patterns):
        return True
    return len(plain) <= _PROMO_MAX_LEN and bool(_PROMO_RE.search(plain))


def _heading_of(plain: str, is_markdown_heading: bool, first_in_doc: bool) -> bool:
    if len(plain) > _HEADING_MAX_LEN:
        return False
    if _CHAPTER_RE.match(plain) or _SPECIAL_HEADING_RE.match(plain):
        return True
    return is_markdown_heading and first_in_doc

# A blank line, but not the form feed used as a document boundary.
_BLANK_LINE_RE = re.compile(r"\n[^\S\f]*\n")

def _is_heading_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > _HEADING_MAX_LEN:
        return False
    if s.startswith("#"):
        return True
    plain = s.replace("*", "")
    return bool(_CHAPTER_RE.match(plain) or _SPECIAL_HEADING_RE.match(plain))

def _blocks(text: str, single_newline: bool) -> list[str]:
    parts = _BLANK_LINE_RE.split(text)
    if single_newline:
        parts = [ln for part in parts for ln in part.split("\n")]
    out: list[str] = []
    for part in parts:
        # A heading directly followed by text (no blank line) is its own block.
        lines = part.split("\n")
        while len(lines) > 1 and _is_heading_line(lines[0]):
            out.append(lines.pop(0))
        out.append("\n".join(lines))
    return out

def clean_and_split_chapters(text: str, cfg: PreprocessConfig | None = None) -> list[Chapter]:
    """Step 2. Clean noise, detect scene breaks, split into chapters."""
    cfg = cfg or PreprocessConfig()
    patterns = [re.compile(p, re.I) for p in (*_NOISE_PATTERNS, *cfg.extra_noise_patterns)]

    chapters: list[Chapter] = []
    current: Chapter | None = None
    pending_break = False
    force_new = False
    seen_content = False

    def start_chapter(title: str | None) -> Chapter:
        nonlocal current
        # Re-use an empty chapter (duplicate title / title page) instead of stacking.
        if current is not None and not current.paragraphs:
            if title:
                current.title = title
            return current
        current = Chapter(index=len(chapters) + 1, title=title)
        chapters.append(current)
        return current

    for block in _blocks(_normalize(text), cfg.single_newline_paragraphs):
        if _DOC_BOUNDARY in block:
            force_new = True
            block = block.replace(_DOC_BOUNDARY, "").strip()
            if not block:
                continue
        if not block.strip():
            continue

        if _SCENE_BREAK_RE.match(block.strip()):
            pending_break = current is not None and bool(current.paragraphs)
            continue

        is_md_heading = block.lstrip().startswith("#")
        para = _collapse(_strip_markdown(block))
        if cfg.drop_noise:
            para = _collapse(_INLINE_NOTE_RE.sub(" ", para))
            if _is_noise(para, patterns):
                continue
        if not para or not any(c.isalnum() for c in para):
            continue

        plain = para.replace("*", "").strip()
        if _heading_of(plain, is_md_heading, first_in_doc=not seen_content):
            start_chapter(plain)
            pending_break = False
            force_new = False
            seen_content = True
            continue
        seen_content = True

        if current is None or force_new:
            start_chapter(None)
            force_new = False
        current.paragraphs.append(
            Paragraph(index=len(current.paragraphs) + 1, text=para, scene_break_before=pending_break)
        )
        pending_break = False

    return [c for c in chapters if c.paragraphs]

# --------------------------------------------------------------------------- #
# Step 3: Corpus A line split (paragraph break + quote boundary)
# --------------------------------------------------------------------------- #

def _italic_mask(par: str) -> tuple[list[bool], list[bool]]:
    """Per-character (is_marker, is_italic). An odd number of '*' means the
    markers are unreliable, so they are treated as plain text-free of italics."""
    n = len(par)
    marker = [False] * n
    italic = [False] * n
    stars = [i for i, ch in enumerate(par) if ch == "*"]
    if not stars or len(stars) % 2:
        if stars:  # unbalanced: still hide the stray markers from the text
            for i in stars:
                marker[i] = True
        return marker, italic
    state = False
    for i, ch in enumerate(par):
        if ch == "*":
            marker[i] = True
            state = not state
        else:
            italic[i] = state
    return marker, italic

def split_paragraph(par: str, cfg: PreprocessConfig | None = None) -> list[tuple[str, bool, bool, int, int]]:
    """Split one paragraph into (text, quoted, italic, start, end) segments."""
    cfg = cfg or PreprocessConfig()
    pairs = cfg.quote_pairs
    marker, italic = _italic_mask(par)
    out: list[tuple[str, bool, bool, int, int]] = []

    def emit(s: int, e: int, quoted: bool) -> None:
        idx = [i for i in range(s, e) if not marker[i]]
        text = "".join(par[i] for i in idx).strip()
        if not text:
            return
        if not quoted and not any(c.isalnum() for c in text):
            return  # stray punctuation between two quotes, e.g. '," he said, "'
        content = [i for i in idx if not par[i].isspace()]
        is_ital = bool(content) and all(italic[i] for i in content)
        # Tighten offsets to the stripped content.
        first = next(i for i in idx if not par[i].isspace())
        last = next(i for i in reversed(idx) if not par[i].isspace())
        out.append((text, quoted, is_ital, first, last + 1))

    closer: str | None = None
    seg_start = 0
    for i, ch in enumerate(par):
        if closer is None:
            if ch in pairs:
                emit(seg_start, i, False)
                closer, seg_start = pairs[ch], i + 1
        elif ch == closer:
            emit(seg_start, i, True)
            closer, seg_start = None, i + 1
    # An unclosed quote runs to the end of the paragraph (multi-paragraph speech).
    emit(seg_start, len(par), closer is not None)
    return out

def split_lines(chapters: list[Chapter], cfg: PreprocessConfig | None = None) -> list[RawLine]:
    """Step 3. Paragraph break + quoted/unquoted boundary -> lines."""
    cfg = cfg or PreprocessConfig()
    lines: list[RawLine] = []
    for ch in chapters:
        for p in ch.paragraphs:
            for k, (text, quoted, ital, s, e) in enumerate(split_paragraph(p.text, cfg)):
                lines.append(
                    RawLine(
                        chapter=ch.index,
                        paragraph=p.index,
                        text=text,
                        quoted=quoted,
                        italic=ital,
                        char_start=s,
                        char_end=e,
                        scene_break_before=p.scene_break_before and k == 0,
                    )
                )
    return lines

# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #

def preprocess(
    source: str | Path, cfg: PreprocessConfig | None = None, *, is_text: bool = False
) -> tuple[list[Chapter], list[RawLine]]:
    """Run Steps 1-3. `source` is a novel folder, a single file, or raw text (is_text=True)."""
    cfg = cfg or PreprocessConfig()
    text = str(source) if is_text else load_source(source)
    chapters = clean_and_split_chapters(text, cfg)
    return chapters, split_lines(chapters, cfg)

if __name__ == "__main__":
    import sys

    chs, lns = preprocess(sys.argv[1])
    print(f"{len(chs)} chapters, {len(lns)} lines")
    for ln in lns[:20]:
        kind = "Q" if ln.quoted else "N"
        print(f"ch{ln.chapter} p{ln.paragraph} [{kind}{'/i' if ln.italic else ''}] {ln.text}")