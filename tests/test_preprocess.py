"""Unit tests for Pipeline 1, Steps 1-3 (src/converter/preprocess.py)."""
import zipfile

import pytest

from src.converter.preprocess import (
    PreprocessConfig,
    book_files,
    load_book,
    clean_and_split_chapters,
    load_raw,
    preprocess,
    split_lines,
    split_paragraph,
)
from src.schema import SourceRef

def run(text, cfg=None):
    return preprocess(text, cfg, is_text=True)

def texts(lines):
    return [ln.text for ln in lines]

# --------------------------------------------------------------------------- #
# Step 1: load_raw
# --------------------------------------------------------------------------- #

def test_load_txt_utf8_bom(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("\ufeffHello \u201cworld\u201d".encode("utf-8"))
    assert load_raw(p) == "Hello \u201cworld\u201d"

def test_load_txt_non_utf8_does_not_crash(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("caf\xe9 au lait, s'il vous pla\xeet".encode("cp1252"))
    assert "caf" in load_raw(p)

def test_load_html_keeps_body_text(tmp_path):
    p = tmp_path / "a.html"
    p.write_text(
        "<html><body><nav>Home | About</nav><article>"
        "<h1>Chapter 1</h1>"
        "<p>Lin Feng drew his sword and stepped into the clearing, where the wind was cold.</p>"
        "<p>He said nothing for a long time, watching the trees move above him.</p>"
        "</article></body></html>",
        encoding="utf-8",
    )
    out = load_raw(p)
    assert "Lin Feng drew his sword" in out

def test_load_epub_reads_spine_in_order(tmp_path):
    p = tmp_path / "book.epub"
    chapter = (
        "<html><body><h1>{t}</h1><p>{body}</p></body></html>"
    )
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        z.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf">'
            "<manifest>"
            '<item id="b" href="b.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="css" href="s.css" media-type="text/css"/>'
            "</manifest>"
            '<spine><itemref idref="a"/><itemref idref="b"/></spine></package>',
        )
        z.writestr("OEBPS/a.xhtml", chapter.format(t="Chapter 1", body="First chapter text here."))
        z.writestr("OEBPS/b.xhtml", chapter.format(t="Chapter 2", body="Second chapter text here."))
    out = load_raw(p)
    assert out.index("First chapter") < out.index("Second chapter")
    chapters, _ = run(out)
    assert len(chapters) == 2

# --------------------------------------------------------------------------- #
# Step 2: cleaning
# --------------------------------------------------------------------------- #

def test_invisible_chars_and_nbsp_removed():
    chapters, lines = run("Hello\u200b\u00a0world.\ufeff")
    assert texts(lines) == ["Hello world."]

def test_crlf_and_hard_wrap_joined_into_one_paragraph():
    chapters, _ = run("Line one\r\nline two\r\n\r\nSecond paragraph.")
    assert [p.text for p in chapters[0].paragraphs] == ["Line one line two", "Second paragraph."]

@pytest.mark.parametrize(
    "noise",
    [
        "A/N: please review",
        "[TN: this is a pun]",
        "Translator's note: see glossary",
        "Translated by SomeGroup",
        "Editor: Someone",
        "Previous Chapter | Next Chapter",
        "Support me on Patreon for more chapters!",
        "\u00a9 2024 Some Author",
    ],
)
def test_noise_paragraphs_dropped(noise):
    _, lines = run(f"The sword fell.\n\n{noise}\n\nHe turned away.")
    assert texts(lines) == ["The sword fell.", "He turned away."]

def test_inline_note_removed():
    _, lines = run("He ate the dumpling [TN: a pun on wealth] quickly.")
    assert texts(lines) == ["He ate the dumpling quickly."]

def test_prose_starting_with_an_is_not_noise():
    _, lines = run("An old man sat by the road.\n\nTranslator was his job title, she said.")
    assert len(lines) == 2

def test_long_paragraph_mentioning_promo_word_kept():
    prose = "He joined the Discord of fighters, " + "and trained every single day " * 10 + "until dawn."
    assert len(prose) > 200
    _, lines = run(prose)
    assert len(lines) == 1

def test_extra_noise_patterns():
    cfg = PreprocessConfig(extra_noise_patterns=(r"^enjoy the chapter",))
    _, lines = run("Enjoy the chapter!\n\nHe ran.", cfg)
    assert texts(lines) == ["He ran."]

def test_drop_noise_can_be_disabled():
    cfg = PreprocessConfig(drop_noise=False)
    _, lines = run("A/N: hello\n\nHe ran.", cfg)
    assert len(lines) == 2

def test_markdown_bold_and_bullets_stripped_italics_kept():
    chapters, _ = run("He was **very** angry.\n\n- a bullet\n\nShe *knew* it.")
    assert [p.text for p in chapters[0].paragraphs] == [
        "He was very angry.",
        "a bullet",
        "She *knew* it.",
    ]

# --------------------------------------------------------------------------- #
# Step 2: scene breaks and chapters
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("brk", ["***", "* * *", "---", "\u25c6\u25c7\u25c6", "~ ~ ~", "###"])
def test_scene_break_flag(brk):
    chapters, lines = run(f"Before the break.\n\n{brk}\n\nAfter the break.")
    paras = chapters[0].paragraphs
    assert len(paras) == 2  # the break itself is not text
    assert paras[0].scene_break_before is False
    assert paras[1].scene_break_before is True
    assert [ln.scene_break_before for ln in lines] == [False, True]

def test_scene_break_before_first_paragraph_ignored():
    chapters, _ = run("***\n\nFirst paragraph.")
    assert chapters[0].paragraphs[0].scene_break_before is False

def test_scene_break_flag_only_on_first_line_of_paragraph():
    _, lines = run("Before.\n\n***\n\n\u201cHi,\u201d he said.")
    flags = [ln.scene_break_before for ln in lines]
    assert flags == [False, True, False]

@pytest.mark.parametrize(
    "heading",
    [
        "Chapter 12: The Duel",
        "Chapter 12",
        "CHAPTER XII - The Duel",
        "Chapter Twelve",
        "Ch. 12 The Duel",
        "Episode 3",
        "# Chapter 7",
        "Prologue",
        "Epilogue: After",
    ],
)
def test_chapter_headings_detected(heading):
    chapters, _ = run(f"Intro text before.\n\n{heading}\n\nBody text after.")
    assert len(chapters) == 2
    assert chapters[1].index == 2
    assert chapters[1].paragraphs[0].text == "Body text after."

def test_chapter_title_kept_and_indices_are_one_based():
    chapters, _ = run("Chapter 1: Start\n\nA.\n\nChapter 2: Next\n\nB.")
    assert [c.index for c in chapters] == [1, 2]
    assert [c.title for c in chapters] == ["Chapter 1: Start", "Chapter 2: Next"]
    assert [p.index for p in chapters[0].paragraphs] == [1]

def test_no_headings_gives_single_untitled_chapter():
    chapters, _ = run("One.\n\nTwo.\n\nThree.")
    assert len(chapters) == 1
    assert chapters[0].title is None
    assert [p.index for p in chapters[0].paragraphs] == [1, 2, 3]

def test_prose_sentence_starting_with_chapter_is_not_a_heading():
    long = "Chapter after chapter he read, and the pages turned while the rain kept falling on the roof " * 2
    chapters, _ = run(f"Start.\n\n{long.strip()}")
    assert len(chapters) == 1

def test_duplicate_heading_does_not_create_empty_chapter():
    chapters, _ = run("# Chapter 1: Start\n\nChapter 1: Start\n\nBody.")
    assert len(chapters) == 1
    assert chapters[0].title == "Chapter 1: Start"

def test_markdown_heading_first_in_doc_becomes_title():
    chapters, _ = run("# The Wandering Sword\n\nBody text here.")
    assert chapters[0].title == "The Wandering Sword"

def test_markdown_heading_later_is_not_a_chapter():
    chapters, _ = run("Chapter 1\n\nBody.\n\n## A subtitle\n\nMore body.")
    assert len(chapters) == 1

def test_heading_directly_followed_by_text_without_blank_line():
    chapters, _ = run("Chapter 1\nHe ran.\n\nShe hid.")
    assert chapters[0].title == "Chapter 1"
    assert [p.text for p in chapters[0].paragraphs] == ["He ran.", "She hid."]

def test_markdown_heading_directly_followed_by_text():
    chapters, _ = run("# Chapter 1\nFirst text.\n\n\f\n\n# Chapter 2\nSecond text.")
    assert [c.title for c in chapters] == ["Chapter 1", "Chapter 2"]
    assert [p.text for p in chapters[1].paragraphs] == ["Second text."]

def test_document_boundary_starts_new_chapter_without_headings():
    chapters, _ = run("First doc.\n\n\f\n\nSecond doc.")
    assert len(chapters) == 2
    assert chapters[1].paragraphs[0].text == "Second doc."

def test_empty_input_gives_nothing():
    chapters, lines = run("")
    assert chapters == [] and lines == []

def test_single_newline_paragraphs_flag():
    text = "First paragraph.\nSecond paragraph.\nThird paragraph."
    default, _ = run(text)
    assert len(default[0].paragraphs) == 1
    cfg = PreprocessConfig(single_newline_paragraphs=True)
    single, _ = run(text, cfg)
    assert len(single[0].paragraphs) == 3

def test_single_newline_mode_still_finds_headings():
    cfg = PreprocessConfig(single_newline_paragraphs=True)
    chapters, _ = run("Chapter 1\nHe ran.\nChapter 2\nShe hid.", cfg)
    assert len(chapters) == 2

# --------------------------------------------------------------------------- #
# Step 3: Corpus A line split
# --------------------------------------------------------------------------- #

def seg(par, cfg=None):
    return [(t, q) for t, q, _, _, _ in split_paragraph(par, cfg)]

def test_curly_quotes_split_quote_and_narration():
    assert seg("\u201cI won't lose,\u201d he said, \u201cnot today.\u201d") == [
        ("I won't lose,", True),
        ("he said,", False),
        ("not today.", True),
    ]

def test_straight_quotes():
    assert seg('"Hello." He waved. "Bye."') == [
        ("Hello.", True),
        ("He waved.", False),
        ("Bye.", True),
    ]

def test_cjk_quotes():
    assert seg("\u300cHmm.\u300d He frowned. \u300eReally?\u300f") == [
        ("Hmm.", True),
        ("He frowned.", False),
        ("Really?", True),
    ]

def test_guillemets():
    assert seg("\u00abBonjour\u00bb, dit-il.") == [("Bonjour", True), (", dit-il.", False)]

def test_apostrophes_do_not_open_quotes():
    assert seg("He didn't know it\u2019s the end.") == [("He didn't know it\u2019s the end.", False)]

def test_unclosed_quote_runs_to_paragraph_end():
    assert seg("\u201cThis speech goes on and on") == [("This speech goes on and on", True)]

def test_stray_closing_quote_is_plain_text():
    assert seg("He said it\u201d and left.") == [("He said it\u201d and left.", False)]

def test_punctuation_only_narration_between_quotes_dropped():
    assert seg('"A." , "B."') == [("A.", True), ("B.", True)]

def test_quoted_ellipsis_is_kept():
    assert seg('"..."') == [("...", True)]

def test_only_narration():
    assert seg("The wind blew.") == [("The wind blew.", False)]

def test_nested_other_quote_style_inside_quote_stays_in_quote():
    assert seg("\u201cHe said \u300cno\u300d to me,\u201d she said.") == [
        ("He said \u300cno\u300d to me,", True),
        ("she said.", False),
    ]

def test_custom_quote_pairs():
    cfg = PreprocessConfig(quote_pairs={"<": ">"})
    assert seg("<Hi> he said.", cfg) == [("Hi", True), ("he said.", False)]

def test_offsets_slice_back_to_text():
    par = "\u201cI won't lose,\u201d he said, \u201cnot today.\u201d"
    for text, _, _, s, e in split_paragraph(par):
        assert par[s:e] == text

def test_offsets_are_into_paragraph_text_with_markers():
    chapters, lines = run("She *knew* it was over. \u201cNo,\u201d she said.")
    par = chapters[0].paragraphs[0].text
    for ln in lines:
        assert par[ln.char_start : ln.char_end].replace("*", "") == ln.text

def test_lines_carry_chapter_and_paragraph_indices():
    _, lines = run("Chapter 1\n\nOne.\n\nTwo.\n\nChapter 2\n\nThree.")
    assert [(ln.chapter, ln.paragraph, ln.text) for ln in lines] == [
        (1, 1, "One."),
        (1, 2, "Two."),
        (2, 1, "Three."),
    ]

def test_each_paragraph_is_split_independently():
    _, lines = run("\u201cHello,\u201d he said.\n\n\u201cHi.\u201d")
    assert [(ln.paragraph, ln.quoted) for ln in lines] == [(1, True), (1, False), (2, True)]

# --- italics --------------------------------------------------------------- #

def test_whole_line_italic_detected_and_markers_stripped():
    _, lines = run("*Why is my heart racing?*")
    assert texts(lines) == ["Why is my heart racing?"]
    assert lines[0].italic is True

def test_partial_italic_not_flagged():
    _, lines = run("He *really* wanted it.")
    assert texts(lines) == ["He really wanted it."]
    assert lines[0].italic is False

def test_italic_wrapping_quotes_marks_quote_italic_only():
    segs = split_paragraph("*\u201cIt can't be,\u201d she thought.*")
    assert [(t, q, i) for t, q, i, _, _ in segs] == [
        ("It can't be,", True, True),
        ("she thought.", False, True),
    ]

def test_unbalanced_asterisk_is_hidden_and_not_italic():
    _, lines = run("He said *hello and left.")
    assert texts(lines) == ["He said hello and left."]
    assert lines[0].italic is False

# --- integration ----------------------------------------------------------- #

def test_end_to_end_sample():
    text = (
        "Translated by SomeGroup\n\n"
        "# Chapter 1: The Duel\n\n"
        "Lin Feng stared at the blade. \u201cI won't lose,\u201d he said.\n\n"
        "A/N: please support me on Patreon\n\n"
        "***\n\n"
        "*Why is my heart racing?*\n\n"
        "Chapter 2: Night\n\n"
        "\u300cHmm.\u300d"
    )
    chapters, lines = run(text)
    assert [c.title for c in chapters] == ["Chapter 1: The Duel", "Chapter 2: Night"]
    assert [(ln.chapter, ln.text, ln.quoted, ln.italic, ln.scene_break_before) for ln in lines] == [
        (1, "Lin Feng stared at the blade.", False, False, False),
        (1, "I won't lose,", True, False, False),
        (1, "he said.", False, False, False),
        (1, "Why is my heart racing?", False, True, True),
        (2, "Hmm.", True, False, False),
    ]

def test_split_lines_matches_preprocess():
    text = "Chapter 1\n\n\u201cHi,\u201d he said."
    chapters = clean_and_split_chapters(text)
    assert split_lines(chapters) == preprocess(text, is_text=True)[1]

def test_source_ref_conversion():
    _, lines = run("Chapter 1\n\nOne.\n\n\u201cTwo.\u201d")
    ref = lines[1].source_ref()
    assert isinstance(ref, SourceRef)
    assert (ref.chapter, ref.paragraph, ref.char_start, ref.char_end) == (1, 2, 1, 5)

# --------------------------------------------------------------------------- #
# Step 1: folder = one novel
# --------------------------------------------------------------------------- #

def make_novel(tmp_path, files):
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path

def test_book_files_natural_order_and_ignores_everything_else(tmp_path):
    make_novel(
        tmp_path,
        {
            "chapter_10.txt": "ten",
            "chapter_2.txt": "two",
            "chapter_1.txt": "one",
            "README.txt": "about this novel",
            "cover.jpg": "not text",
            "notes.json": "{}",
            ".hidden.txt": "secret",
        },
    )
    (tmp_path / "extras").mkdir()
    (tmp_path / "extras" / "chapter_0.txt").write_text("nested")
    assert [f.name for f in book_files(tmp_path)] == ["chapter_1.txt", "chapter_2.txt", "chapter_10.txt"]

def test_load_book_joins_files_with_boundary(tmp_path):
    make_novel(tmp_path, {"001.txt": "First file.", "002.txt": "Second file."})
    out = load_book(tmp_path)
    assert out.index("First file.") < out.index("Second file.")
    assert "\f" in out

def test_each_file_is_a_chapter_even_without_headings(tmp_path):
    make_novel(tmp_path, {"001.txt": "Alpha.\n\nBeta.", "002.txt": "Gamma."})
    chapters, lines = preprocess(tmp_path)
    assert [len(c.paragraphs) for c in chapters] == [2, 1]
    assert [(ln.chapter, ln.text) for ln in lines] == [(1, "Alpha."), (1, "Beta."), (2, "Gamma.")]

def test_file_heading_is_used_and_not_duplicated(tmp_path):
    make_novel(
        tmp_path,
        {"001.txt": "Chapter 1: Start\n\nAlpha.", "002.txt": "Chapter 2: Next\n\nBeta."},
    )
    chapters, _ = preprocess(tmp_path)
    assert [c.title for c in chapters] == ["Chapter 1: Start", "Chapter 2: Next"]

def test_load_book_skips_empty_files(tmp_path):
    make_novel(tmp_path, {"001.txt": "Alpha.", "002.txt": "   \n", "003.txt": "Gamma."})
    chapters, _ = preprocess(tmp_path)
    assert len(chapters) == 2

def test_load_book_empty_folder_raises(tmp_path):
    make_novel(tmp_path, {"cover.jpg": "x", "README.txt": "y"})
    with pytest.raises(ValueError):
        load_book(tmp_path)

def test_preprocess_accepts_single_file_path(tmp_path):
    f = tmp_path / "book.txt"
    f.write_text("Chapter 1\n\nHello there.", encoding="utf-8")
    chapters, lines = preprocess(f)
    assert len(chapters) == 1 and texts(lines) == ["Hello there."]