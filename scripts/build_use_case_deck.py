#!/usr/bin/env python3
"""
Generate the Personal AI Framework (Skippy) use-case architecture deck.

Audience: a work colleague evaluating an edge NPU. They already understand
AI/ML; they don't care about UI polish. They want to know (1) what work the
AI/ML pipeline actually does, (2) how data flows through the system, (3) the
performance KPIs, (4) how it would perform on a target NPU.

Output: docs/personal-ai-use-cases.pptx (gitignored, private).

Deck template and helpers are ported from keyhole's build_deck.py so the
Skippy deck shares the same visual identity and can feed the same
pptx_template_converter corporate-template grafting pipeline.

Corporate-template pipeline (MERGE_TARGET mode):
  SKIPPY_DECK_MERGE_TARGET=1 python3 scripts/build_use_case_deck.py
  → outputs a "canvas" deck with no branding or background; the
    pptx_template_converter can then graft an arbitrary corporate
    template onto it via color-slot remapping.
"""
import io
import os
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "personal-ai-use-cases.pptx"


# ============================================================
# Color Palette — ported from keyhole/scripts/build_deck.py
# ============================================================

class C:
    """Consistent color palette across all slides (from keyhole template)."""
    BG_DARK = RGBColor(0x1A, 0x1A, 0x2E)
    BG_SLIDE = RGBColor(0x16, 0x21, 0x3E)
    ACCENT_BLUE = RGBColor(0x00, 0xD4, 0xFF)
    ACCENT_GREEN = RGBColor(0x00, 0xFF, 0x88)
    ACCENT_ORANGE = RGBColor(0xFF, 0x8C, 0x00)
    ACCENT_RED = RGBColor(0xFF, 0x44, 0x44)
    ACCENT_PURPLE = RGBColor(0xBB, 0x86, 0xFC)
    ACCENT_AMBER = RGBColor(0xF5, 0x9E, 0x0B)    # storage / projected
    ACCENT_INDIGO = RGBColor(0x63, 0x66, 0xF1)   # transport / emphasis
    TEXT_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    TEXT_DIM = RGBColor(0xAA, 0xAA, 0xCC)
    TEXT_BRIGHT = RGBColor(0xE0, 0xE0, 0xFF)
    TABLE_HEADER = RGBColor(0x0D, 0x47, 0xA1)
    TABLE_ROW_1 = RGBColor(0x1A, 0x23, 0x40)
    TABLE_ROW_2 = RGBColor(0x22, 0x2B, 0x4A)
    FEASIBLE = RGBColor(0x00, 0xE6, 0x76)
    NOT_FEASIBLE = RGBColor(0xFF, 0x44, 0x44)


# Legacy aliases — existing Skippy slide functions reference these names.
# Mapped into the new palette so slide content renders with keyhole's
# visual identity without rewriting every slide.
NAVY = C.BG_DARK
INK = C.BG_DARK
SURFACE = C.TABLE_ROW_2
ACCENT = C.ACCENT_BLUE             # primary accent — cyan (keyhole identity)
ACCENT2 = C.ACCENT_GREEN           # green (measured)
ACCENT3 = C.ACCENT_AMBER           # amber (projected)
ACCENT4 = C.ACCENT_RED             # red (gap)
# TEXT maps to the off-white TEXT_BRIGHT (E0E0FF) rather than pure white
# (FFFFFF). Crucial for the MERGE_TARGET pipeline: the color map sends
# E0E0FF → dk1 (dark text on light bg), while FFFFFF → lt1 (white on
# white, invisible). Matches keyhole's convention — their add_bullet_box
# also uses TEXT_BRIGHT for body content for the same reason.
TEXT = C.TEXT_BRIGHT
MUTED = C.TEXT_DIM
FAINT = C.TABLE_ROW_1


# ============================================================
# Widescreen layout constants (from keyhole template)
# ============================================================

SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5
SLIDE_W = Inches(SLIDE_W_IN)
SLIDE_H = Inches(SLIDE_H_IN)
CONTENT_LEFT = 0.5
CONTENT_W = 12.3
TITLE_TOP = 0.2
SUBTITLE_TOP = 0.8
CONTENT_TOP = 1.4
FOOTER_TOP = 7.1
PROJECT_FOOTER = "Personal AI Framework (Skippy)  •  v5.9 architecture"


# MERGE_TARGET mode: when SKIPPY_DECK_MERGE_TARGET=1 (or its keyhole
# equivalent), the deck is built for pptx_template_converter — no
# background, no top accent stripe, no footer (the corporate template
# provides all branding via masters). See keyhole/scripts/build_deck.py
# for the canonical implementation.
MERGE_TARGET = (
    os.environ.get("SKIPPY_DECK_MERGE_TARGET", "").strip() == "1"
    or os.environ.get("KEYHOLE_DECK_MERGE_TARGET", "").strip() == "1"
)


prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H


# ============================================================
# Keyhole-template helpers (verbatim from keyhole/scripts/build_deck.py)
# ============================================================

def set_slide_bg(slide, color=C.BG_SLIDE):
    """Set slide background to a solid color."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text,
                 font_size=18, color=C.TEXT_WHITE, bold=False,
                 alignment=PP_ALIGN.LEFT, font_name="Segoe UI"):
    """Add a styled text box using run-level formatting.

    Writes formatting to the run-level <a:rPr> (via explicit Run) rather
    than the paragraph default <a:pPr><a:defRPr>. PowerPoint's Font Color
    picker edits run-level properties — without <a:rPr> on the run, user
    edits sometimes fail to apply.
    """
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = alignment
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = font_name
    return txBox


def add_title_bar(slide, title, subtitle=None):
    """Keyhole-style title bar — 28pt accent-blue title + short accent
    underline bar. Used for the hero/section slides.
    """
    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(9), Inches(0.6),
                 title, font_size=28, color=C.ACCENT_BLUE, bold=True)
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.5), Inches(0.85), Inches(2), Pt(3),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = C.ACCENT_BLUE
    line.line.fill.background()
    if subtitle:
        add_text_box(slide, Inches(0.5), Inches(0.95), Inches(9), Inches(0.4),
                     subtitle, font_size=14, color=C.TEXT_DIM)


def add_title_subtitle(slide, title, subtitle=None):
    """Compact title (22pt) + subtitle (13pt) pair. The default title
    style for content slides in the keyhole template."""
    add_text_box(slide, Inches(CONTENT_LEFT), Inches(TITLE_TOP),
                 Inches(CONTENT_W), Inches(0.6),
                 title, font_size=22, color=C.ACCENT_BLUE, bold=True)
    if subtitle:
        add_text_box(slide, Inches(CONTENT_LEFT), Inches(SUBTITLE_TOP),
                     Inches(CONTENT_W), Inches(0.4),
                     subtitle, font_size=13, color=C.TEXT_DIM)


def add_bullet_box(slide, left, top, width, height, items,
                   font_size=13, font_name="Segoe UI"):
    """Keyhole bullet-box pattern. items is list of:
      - "" / None: blank separator line
      - str: regular bullet at TEXT_BRIGHT
      - (text, color): colored regular line
      - (text, color, bold): colored bold line
    Coordinates are in Inches (plain floats, not Inches()-wrapped).
    """
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height),
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if item is None or item == "":
            text, color, bold = "", C.TEXT_DIM, False
        elif isinstance(item, str):
            text, color, bold = item, C.TEXT_BRIGHT, False
        else:
            text = item[0]
            color = item[1] if len(item) > 1 else C.TEXT_BRIGHT
            bold = item[2] if len(item) > 2 else False
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        if p.runs:
            r = p.runs[0]
            r.font.size = Pt(font_size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.name = font_name
    return txBox


def add_pipeline_strip(slide, stages, top_in=1.2, accent_color=None):
    """Compact pipeline state strip — each stage is a rounded box with
    an arrow between, highlighted boxes call out "what changed on this
    slide". stages is list of str OR (label, highlighted_bool) tuples.
    """
    accent = accent_color if accent_color is not None else C.ACCENT_INDIGO
    total_w = CONTENT_W
    n = len(stages)
    gap = 0.12
    arrow_w = 0.22
    box_w = (total_w - (n - 1) * (gap + arrow_w)) / n
    box_h = 0.55
    x = CONTENT_LEFT

    for i, entry in enumerate(stages):
        if isinstance(entry, tuple):
            label, highlighted = entry
        else:
            label, highlighted = entry, False
        shp = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(top_in), Inches(box_w), Inches(box_h),
        )
        if highlighted:
            shp.fill.solid(); shp.fill.fore_color.rgb = accent
            shp.line.color.rgb = accent
            text_color = C.TEXT_WHITE
            bold = True
        else:
            shp.fill.solid(); shp.fill.fore_color.rgb = C.TABLE_ROW_2
            shp.line.color.rgb = C.TEXT_DIM
            text_color = C.TEXT_DIM
            bold = False
        shp.line.width = Pt(1.0)
        tf = shp.text_frame
        tf.word_wrap = True
        tf.margin_left = Emu(40000); tf.margin_right = Emu(40000)
        tf.margin_top = Emu(20000); tf.margin_bottom = Emu(20000)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.size = Pt(10)
        r.font.bold = bold
        r.font.color.rgb = text_color
        r.font.name = "Segoe UI"
        if i < n - 1:
            ar_x = x + box_w + gap / 2
            ar = add_text_box(slide, Inches(ar_x), Inches(top_in),
                              Inches(arrow_w), Inches(box_h),
                              "→", font_size=14, color=C.TEXT_DIM,
                              alignment=PP_ALIGN.CENTER)
            ar.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        x += box_w + gap + arrow_w


def add_styled_table(slide, left, top, width, height,
                     headers, rows, col_widths=None, highlight_rows=None,
                     font_size=10, header_font_size=11):
    """Keyhole-style dark-themed table. highlight_rows is iterable of
    1-based indices to emphasize with indigo fill + bold white text."""
    num_rows = len(rows) + 1
    num_cols = len(headers)
    highlight = set(highlight_rows or [])
    table_shape = slide.shapes.add_table(
        num_rows, num_cols, left, top, width, height
    )
    table = table_shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w

    def _set_cell(cell, text, *, sz, color, bold, align=PP_ALIGN.CENTER):
        cell.text = ""
        tf = cell.text_frame
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = str(text)
        run.font.size = Pt(sz)
        run.font.color.rgb = color
        run.font.bold = bold
        run.font.name = "Segoe UI"

    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C.TABLE_HEADER
        _set_cell(cell, header, sz=header_font_size,
                  color=C.TEXT_WHITE, bold=True)
    for r_idx, row in enumerate(rows):
        row_idx = r_idx + 1
        is_hl = row_idx in highlight
        bg = C.ACCENT_INDIGO if is_hl else (C.TABLE_ROW_1 if r_idx % 2 == 0 else C.TABLE_ROW_2)
        fg = C.TEXT_WHITE if is_hl else C.TEXT_BRIGHT
        for c_idx, value in enumerate(row):
            cell = table.cell(row_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            _set_cell(cell, value, sz=font_size, color=fg, bold=is_hl)
    return table_shape


def new_slide(bg_color=None, accent_stripe=True):
    """Keyhole-style new blank slide — solid background + top accent
    stripe. Skipped in MERGE_TARGET mode so corporate template masters
    own the branding. Operates on the module-level `prs`.

    NOTE: final branded deck is produced by piping this output through
    pptx_template_converter/convert.py — this builder emits the plain
    dark-synthwave source deck.
    """
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if not MERGE_TARGET:
        set_slide_bg(slide, bg_color or C.BG_SLIDE)
        if accent_stripe:
            stripe = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(0), Inches(0), SLIDE_W, Pt(4),
            )
            stripe.fill.solid()
            stripe.fill.fore_color.rgb = C.ACCENT_BLUE
            stripe.line.fill.background()
    return slide


def finalize_footers():
    """Add footers after all slides are built (so we know the total).
    Replaces per-slide manual add_footer() calls with a single final sweep."""
    if MERGE_TARGET:
        return  # corporate template master owns the footer
    total = len(prs.slides)
    for i, slide in enumerate(prs.slides):
        add_text_box(slide, Inches(CONTENT_LEFT), Inches(FOOTER_TOP),
                     Inches(CONTENT_W), Inches(0.3),
                     f"{PROJECT_FOOTER}  •  {i + 1}/{total}",
                     font_size=9, color=C.TEXT_DIM,
                     alignment=PP_ALIGN.RIGHT)


def fig_to_image_stream(fig):
    """Convert a matplotlib figure to a BytesIO PNG stream. Requires
    `import matplotlib.pyplot as plt` in the caller. Kept here so chart-
    bearing slides can be added later without restructuring helpers.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================
# Skippy compatibility shims — existing slide functions (slide1_title,
# slide2_exec_summary, ... slide_skippy_moe_upgrade) reference these
# original names with specific signatures. Map to the keyhole primitives
# above so slide content renders with the new template.
# ============================================================

def add_blank():
    """Legacy entry — equivalent to new_slide() with keyhole defaults."""
    return new_slide()


def add_text(slide, x, y, w, h, text, *, size=14, bold=False, color=TEXT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, mono=False):
    """Multi-line text box with Skippy's richer arg set (mono, anchor,
    multi-line). Kept distinct from keyhole's add_text_box because
    several Skippy slides rely on multi-line splits + mono font."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Consolas" if mono else "Segoe UI"
    return tb


def add_title(slide, title, subtitle=None):
    """Legacy Skippy title bar. Redirected to the keyhole compact
    title+subtitle so all slides share the template's title identity
    (22pt accent-blue, 13pt dim subtitle). The prior full-width ACCENT
    stripe is now provided by `new_slide`'s top accent stripe."""
    add_title_subtitle(slide, title, subtitle)


def add_box(slide, x, y, w, h, text, *, fill=SURFACE, border=ACCENT,
            size=12, bold=False, color=TEXT):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = border; shp.line.width = Pt(1.25)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(80000); tf.margin_right = Emu(80000)
    tf.margin_top = Emu(60000); tf.margin_bottom = Emu(60000)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold and i == 0
        run.font.color.rgb = color
    return shp


def add_arrow(slide, x1, y1, x2, y2, color=ACCENT):
    ln = slide.shapes.add_connector(2, x1, y1, x2, y2)  # STRAIGHT
    ln.line.color.rgb = color
    ln.line.width = Pt(2)
    # Arrowhead
    fmt = ln.line
    fmt.fill.solid(); fmt.fill.fore_color.rgb = color
    xml = ln._element
    # Add arrow end element via raw XML
    from pptx.oxml.ns import qn
    from lxml import etree
    lnEl = xml.xpath(".//a:ln", namespaces={"a":"http://schemas.openxmlformats.org/drawingml/2006/main"})[0]
    tail = etree.SubElement(lnEl, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med"); tail.set("len", "med")


def add_bullets(slide, x, y, w, h, items, size=14, indent=0):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.level = indent
        run = p.add_run()
        run.text = "• " + item
        run.font.size = Pt(size)
        run.font.color.rgb = TEXT
        p.space_after = Pt(4)
    return tb


def add_table(slide, x, y, w, h, headers, rows, *, header_fill=ACCENT,
              header_color=RGBColor(0xFF,0xFF,0xFF), band_fill=FAINT,
              font_size=11, highlight_rows=None):
    highlight_rows = highlight_rows or {}
    cols = len(headers)
    n_rows = len(rows) + 1
    tbl_shape = slide.shapes.add_table(n_rows, cols, x, y, w, h)
    tbl = tbl_shape.table
    # Header
    for c, h_text in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.text = ""
        cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
        cell.margin_left = Emu(40000); cell.margin_right = Emu(40000)
        cell.margin_top = Emu(20000); cell.margin_bottom = Emu(20000)
        tf = cell.text_frame
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = h_text
        r.font.size = Pt(font_size); r.font.bold = True; r.font.color.rgb = header_color
    # Rows
    for ri, row in enumerate(rows, start=1):
        fill = header_fill if ri in highlight_rows else (band_fill if ri % 2 == 0 else SURFACE)
        tcolor = RGBColor(0xFF,0xFF,0xFF) if ri in highlight_rows else TEXT
        for c, val in enumerate(row):
            cell = tbl.cell(ri, c)
            cell.text = ""
            cell.fill.solid(); cell.fill.fore_color.rgb = fill
            cell.margin_left = Emu(40000); cell.margin_right = Emu(40000)
            cell.margin_top = Emu(20000); cell.margin_bottom = Emu(20000)
            tf = cell.text_frame
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            r = p.add_run(); r.text = str(val)
            r.font.size = Pt(font_size); r.font.color.rgb = tcolor
            if ri in highlight_rows:
                r.font.bold = True
    return tbl


def add_footer(slide, n, total):
    add_text(slide, Inches(0.5), Inches(7.1), Inches(12),  Inches(0.3),
             f"Personal AI Framework (Skippy)  •  v5.9 architecture  •  {n}/{total}",
             size=9, color=MUTED, align=PP_ALIGN.RIGHT)


# ── Content ──

SLIDES = []


def slide1_title():
    s = add_blank()
    add_text(s, Inches(0.5), Inches(2.4), Inches(12.3), Inches(1.2),
             "Personal AI Framework", size=48, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.5), Inches(3.5), Inches(12.3), Inches(0.7),
             "A fully-local AI assistant — use-case architecture",
             size=22, color=MUTED, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.5), Inches(4.4), Inches(12.3), Inches(0.5),
             "Qwen 2.5 7B v4 (fine-tuned, production)  |  61K-doc RAG  |  hybrid retrieval  |  tool-using agent  |  recipe-template PoC",
             size=14, color=ACCENT, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.5), Inches(6.3), Inches(12.3), Inches(0.5),
             "Measured on RTX 5090 @ 32GB  •  Projected to 200-TOPS edge NPU",
             size=12, color=MUTED, align=PP_ALIGN.CENTER)
SLIDES.append(slide1_title)


def slide2_exec_summary():
    s = add_blank()
    add_title(s, "Executive summary",
              "What the framework does and why it matters for edge")
    add_bullets(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(5.4), [
        "Private AI assistant — runs 100% locally, no cloud API calls.",
        "Answers questions using your own corpus (61K+ documents: emails, transcripts, PDFs, source code).",
        "Retrieves context via hybrid search (semantic + BM25 + reranking), injects into Qwen 2.5 7B v4 (fine-tuned, production).",
        "Agentic: can call tools (read files, web search, send Gmail, create calendar events, schedule reminders, run scripts).",
        "Learns from use — 👍/👎 feedback excludes bad turns; LoRA retrains on your writing style.",
        "Multi-user with per-user data isolation (bcrypt auth, per-user SQLite + ChromaDB collections).",
        "Observable — Prometheus /metrics + Grafana dashboard for TTFT, throughput, RAG hits, tool calls.",
        "Why this deck matters: data flow + KPIs let us size a 200-TOPS edge NPU replacement for the RTX 5090 today.",
    ], size=14)
SLIDES.append(slide2_exec_summary)


def slide3_block_diagram():
    s = add_blank()
    add_title(s, "System block diagram",
              "Major components and the data paths between them")

    # Row 1: Input sources
    L = Inches(0.5); TOP = Inches(1.3); BW = Inches(2.2); BH = Inches(0.8); GAP = Inches(0.25)
    add_box(s, L, TOP, BW, BH, "Web UI\n(chat + voice)", fill=SURFACE, border=ACCENT)
    add_box(s, L + BW + GAP, TOP, BW, BH, "Upload\n(docs / audio / OCR)", fill=SURFACE, border=ACCENT)
    add_box(s, L + (BW+GAP)*2, TOP, BW, BH, "Mic (Whisper STT)\nin-browser", fill=SURFACE, border=ACCENT)
    add_box(s, L + (BW+GAP)*3, TOP, BW, BH, "Mobile / LAN\n(phone via IP)", fill=SURFACE, border=ACCENT)

    # Row 2: LLM Server (FastAPI) - spans
    LT = Inches(2.05)
    srv = add_box(s, L, LT, Inches(12.3), Inches(0.65),
                  "FastAPI LLM Server (auth middleware · per-user context · /metrics)",
                  fill=ACCENT, border=ACCENT, color=RGBColor(0xFF,0xFF,0xFF), bold=True, size=13)

    # Row 3: AI/ML subsystems
    T3 = Inches(2.85); H3 = Inches(1.7)
    add_box(s, L, T3, BW, H3,
            "Tool-detection\npass\n\n(short prompt,\nJSON out)",
            fill=SURFACE, border=ACCENT2, size=11)
    add_box(s, L + (BW+GAP), T3, BW, H3,
            "RAG retrieval\n\nChromaDB +\nBM25 + rerank\n(hybrid)",
            fill=SURFACE, border=ACCENT2, size=11)
    add_box(s, L + (BW+GAP)*2, T3, BW, H3,
            "Main LLM\ninference\n\nQwen 2.5 7B v4\nQ4_K_M GGUF\n(llama.cpp, CUDA)",
            fill=ACCENT2, border=ACCENT2, color=RGBColor(0x0F,0x19,0x2E), size=11, bold=True)
    add_box(s, L + (BW+GAP)*3, T3, BW, H3,
            "Memory\n(conversation)\n\nper-user\nChromaDB\ncollection",
            fill=SURFACE, border=ACCENT2, size=11)
    add_box(s, L + (BW+GAP)*4, T3, BW, H3,
            "Learned facts\n\nper-user\nChromaDB\ncollection",
            fill=SURFACE, border=ACCENT2, size=11)

    # Row 4: Storage + training
    T4 = Inches(4.75); H4 = Inches(1.3)
    add_box(s, L, T4, Inches(3.0), H4,
            "SQLite (per user)\nconversations · reminders · feedback (👍/👎) · settings",
            fill=SURFACE, border=ACCENT3, size=11)
    add_box(s, L + Inches(3.2), T4, Inches(3.0), H4,
            "Tool executor\nwrite_file · send_email · create_event · run_script · web_search · schedule_reminder",
            fill=SURFACE, border=ACCENT3, size=10)
    add_box(s, L + Inches(6.4), T4, Inches(3.0), H4,
            "LoRA training\nPEFT (rank 64) → GGUF merge → deploy\nSelective: skip 👎 and excluded convos",
            fill=SURFACE, border=ACCENT3, size=11)
    add_box(s, L + Inches(9.6), T4, Inches(3.2), H4,
            "Prometheus /metrics\nTTFT · tok/s · tool calls · 👍/👎 · RAG docs",
            fill=SURFACE, border=ACCENT3, size=11)

    # Legend
    add_text(s, Inches(0.5), Inches(6.25), Inches(12.3), Inches(0.3),
             "Green = AI/ML inference  |  Amber = storage/data  |  Indigo = transport",
             size=11, color=MUTED, align=PP_ALIGN.CENTER)
SLIDES.append(slide3_block_diagram)


def slide4_inference_path():
    s = add_blank()
    add_title(s, "Data flow — single-turn inference (RAG)",
              "What happens between 'Enter' and the first token")

    steps = [
        ("User prompt", SURFACE, ACCENT),
        ("Query rewrite\n(LLM, 64 tok)", SURFACE, ACCENT2),
        ("Hybrid retrieval\nChromaDB + BM25 + rerank", SURFACE, ACCENT2),
        ("Top-k chunks\n(semantic + BM25 merged,\nreranked)", SURFACE, ACCENT2),
        ("Assemble prompt\nsystem + facts + memory\n+ RAG + history + user", SURFACE, ACCENT3),
        ("Main LLM inference\nQwen 2.5 7B v4\n(streaming)", ACCENT2, ACCENT2),
        ("Stream tokens\n→ Web UI (SSE)\n→ metrics recorded", SURFACE, ACCENT),
    ]

    L = Inches(0.5); T = Inches(1.6); BW = Inches(1.7); BH = Inches(1.4); GAP = Inches(0.1)
    for i, (txt, fill, border) in enumerate(steps):
        x = L + (BW + GAP) * i
        is_llm = fill == ACCENT2
        add_box(s, x, T, BW, BH, txt, fill=fill, border=border,
                size=10, bold=is_llm,
                color=RGBColor(0x0F,0x19,0x2E) if is_llm else TEXT)

    # KPIs below
    T2 = Inches(3.4)
    kpi_box = add_box(s, L, T2, Inches(12.3), Inches(1.0),
                      "Measured on RTX 5090  ·  Qwen 2.5 7B v4 Q4_K_M  ·  16K ctx",
                      fill=INK, border=ACCENT, size=12, bold=True)
    add_text(s, L, Inches(4.5), Inches(12.3), Inches(2.6), [
        "• Query rewrite: 20–40 ms (small LLM call with 64 max tokens)",
        "• Hybrid retrieval: 60–120 ms (BM25 index + ChromaDB HNSW + cross-encoder reranker on top-20)",
        "• Prompt assembly: < 5 ms (pure Python string building)",
        "• TTFT (time to first token): 30–120 ms — 7B prefill is faster than 14B (smaller weight read)",
        "• Throughput: 180–215 tok/s sustained — 7B Q4 ~4.7 GB weights/token vs 14B's 8.7 GB",
        "• End-to-end for a 200-token answer: typically 1.0–2.0 s wall clock",
    ], size=13)
SLIDES.append(slide4_inference_path)


def slide5_agent_path():
    s = add_blank()
    add_title(s, "Data flow — agent tool pipeline",
              "Two-pass LLM: detect → (safe tools loop) → approve → execute → final generation")

    # Column 1: detection pass
    add_box(s, Inches(0.5), Inches(1.5), Inches(3.6), Inches(1.3),
            "Pass 1: Tool detection\nShort system prompt lists tools + few-shot examples.\nLLM emits <tool>...</tool> or nothing.\nmax_tokens=2048",
            fill=SURFACE, border=ACCENT2, size=11)
    # Column 2: loop
    add_box(s, Inches(0.5), Inches(3.0), Inches(3.6), Inches(2.3),
            "Pass 2a: Safe-tool loop (≤ 5 iterations)\n\nread_file · list_files · web_search · git_status · list_calendar_events\n\nEach result is fed back into detection prompt as <tool_output>…</tool_output>.\nLoop stops when the LLM emits no tool call.",
            fill=SURFACE, border=ACCENT2, size=11)
    # Column 3: confirm
    add_box(s, Inches(4.3), Inches(1.5), Inches(3.6), Inches(2.0),
            "Confirm tools\nwrite_file · run_script · send_email · create_calendar_event · schedule_reminder\n\nBreak the loop → surface pending-action card → wait for user approve/deny.",
            fill=SURFACE, border=ACCENT3, size=11)
    # Column 4: final gen
    add_box(s, Inches(4.3), Inches(3.7), Inches(3.6), Inches(1.6),
            "Pass 2b: Final generation\nAll tool results wrapped in <toolname_output>…</toolname_output> tags + scrubber to strip leaked scaffolding.",
            fill=ACCENT2, border=ACCENT2, color=RGBColor(0x0F,0x19,0x2E), size=11, bold=True)
    # Column 5: UI
    add_box(s, Inches(8.1), Inches(1.5), Inches(4.7), Inches(3.8),
            "UI effects\n• Collapsible 🔧 trace card per step\n• Pending-action card (param preview + diff)\n• Overwrite warning for existing files\n• 'Save as copy' sibling path option\n• Final streaming answer below traces",
            fill=SURFACE, border=ACCENT, size=11)

    add_text(s, Inches(0.5), Inches(5.7), Inches(12.3), Inches(1.7), [
        "AI/ML cost breakdown (per user-visible turn, worst case):",
        "• 1 × detection pass (fast, short output)",
        "• ≤ 5 × safe-tool detections + executions (each ≈ 1 detection pass + 1 sandboxed exec)",
        "• 1 × RAG retrieval (reused from inference path)",
        "• 1 × final generation (full-length answer)",
        "Observed: adds 200–800 ms overhead on tool-using turns; tool execution itself is usually < 50 ms.",
    ], size=12)
SLIDES.append(slide5_agent_path)


def slide6_memory_rlhf():
    s = add_blank()
    add_title(s, "Data flow — memory + RLHF feedback loop",
              "How conversations flow into long-term memory and back into the model")

    # Flow: chat → SQLite → (auto) ingest → ChromaDB → retrieved as memory
    L = Inches(0.5); T = Inches(1.5); BW = Inches(2.0); BH = Inches(1.2); GAP = Inches(0.3)
    add_box(s, L, T, BW, BH, "Chat turn\n(user + assistant)", fill=SURFACE, border=ACCENT)
    add_box(s, L + (BW+GAP), T, BW, BH, "SQLite\nconversations.db\n(per-user)", fill=SURFACE, border=ACCENT3)
    add_box(s, L + (BW+GAP)*2, T, BW, BH, "Auto-ingest\n(on toggle)", fill=SURFACE, border=ACCENT2)
    add_box(s, L + (BW+GAP)*3, T, BW, BH, "ChromaDB\nmemory_<user>\ncollection", fill=SURFACE, border=ACCENT2)
    add_box(s, L + (BW+GAP)*4, T, BW, BH, "Next retrieval\nmemory hits\ninjected into ctx", fill=ACCENT2, border=ACCENT2, color=RGBColor(0x0F,0x19,0x2E), bold=True)

    # RLHF leg
    T2 = Inches(3.2)
    add_box(s, L, T2, BW, BH, "User rates\n👍 / 👎", fill=SURFACE, border=ACCENT)
    add_box(s, L + (BW+GAP), T2, BW, BH, "feedback table\n(in conversations.db)", fill=SURFACE, border=ACCENT3)
    add_box(s, L + (BW+GAP)*2, T2, BW, BH, "collect_training_data.py\nskips 👎 pairs\nskips excluded convos", fill=SURFACE, border=ACCENT2)
    add_box(s, L + (BW+GAP)*3, T2, BW, BH, "LoRA training\nPEFT rank 64\nbf16 · 3 epochs", fill=SURFACE, border=ACCENT2)
    add_box(s, L + (BW+GAP)*4, T2, BW, BH, "GGUF merge\n→ active model\nauto-restore", fill=ACCENT2, border=ACCENT2, color=RGBColor(0x0F,0x19,0x2E), bold=True)

    add_text(s, Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.2), [
        "Why this matters for edge sizing:",
        "• Inference and memory retrieval are hot paths — every turn. These must run at interactive speed (< 3 s end-to-end).",
        "• Training is a background path — runs when the user triggers it, or on schedule. Can tolerate 2–3 h on device or offload to host.",
        "• RAG embeddings (ChromaDB) are cold storage — CPU-bound, not NPU-bound. 61K-doc index uses ~600 MB RAM + ~250 MB disk.",
        "• LoRA retraining is the NPU's stress-test case: Qwen 2.5 14B bf16 + rank-64 adapter ≈ 28 GB peak; usually offloaded on edge.",
    ], size=13)
SLIDES.append(slide6_memory_rlhf)


def slide7_kpis():
    s = add_blank()
    add_title(s, "Measured KPIs — what the RTX 5090 actually does",
              "Per-turn observability from Prometheus /metrics (v5.8+)")

    rows = [
        ("Time to first token (TTFT)",        "40–170 ms",       "p50 / p95",    "dominated by prompt prefill + KV warmup"),
        ("Throughput (streaming)",             "85–140 tok/s",     "sustained",    "bandwidth-bound at Q4_K_M · 9 GB model"),
        ("End-to-end, 200-token answer",       "1.5–3.0 s",         "wall",         "includes RAG + tool pass + generation"),
        ("RAG retrieval",                      "60–120 ms",         "k=5",          "hybrid (BM25 + HNSW + rerank top-20)"),
        ("Tool detection pass",                 "80–150 ms",         "64–2048 tok",  "separate short LLM call"),
        ("Tool execution (safe tools)",        "5–50 ms",           "local only",    "web_search: 300–800 ms (network)"),
        ("Per-user memory retrieval",          "40–80 ms",          "top-2",        "ChromaDB collection per user"),
        ("Peak VRAM (14B Q4_K_M, 16K ctx)",     "≈ 9.2 GB",           "steady",       "KV cache grows ~0.5 GB per 1K tokens"),
        ("Host CPU during inference",          "< 10%",              "single core",  "llama.cpp offloads all layers to GPU"),
        ("Docs in knowledge base",              "61,500+",            "ChromaDB",     "~600 MB RAM, 250 MB disk"),
        ("Model load time (cold)",             "20–30 s",            "GGUF mmap",    "one-time; hot restart re-uses page cache"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(5.0),
              ["Metric", "Measured", "How", "Notes"],
              rows, font_size=11)
    add_text(s, Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.4),
             "All numbers from skippy_* Prometheus histograms on production usage.",
             size=11, color=MUTED, align=PP_ALIGN.CENTER)
SLIDES.append(slide7_kpis)


def slide8_bw_math():
    s = add_blank()
    add_title(s, "Memory bandwidth — the first-order edge constraint",
              "LLM inference is memory-bandwidth-bound, not compute-bound")

    add_box(s, Inches(0.5), Inches(1.4), Inches(6.2), Inches(1.4),
            "For dense autoregressive models,\nevery token decode reads the full weight set at least once.\n\nBW_needed (GB/s) ≈ Model_size (GB) × tok/s_desired",
            fill=SURFACE, border=ACCENT2, size=14, bold=True)

    add_box(s, Inches(6.9), Inches(1.4), Inches(6.0), Inches(1.4),
            "Consequences for edge:\n• Big model = low tok/s unless BW is huge\n• Quantization (Q4 / INT4) halves the BW need\n• MoE only loads experts — lowers effective model size, but needs smart caching",
            fill=SURFACE, border=ACCENT3, size=12)

    # BW of various systems — incl. memory upgrade ladder for the target NPU.
    rows = [
        ("Vendor RTX 5090 (384-bit GDDR7)",         "1792 GB/s",    "reference",   "Measured"),
        ("Apple M3 Max (512-bit LPDDR5)",            "400 GB/s",     "reference",   "Measured"),
        ("128-bit LPDDR5X @ 8.4 GT/s — peak",        "134.4 GB/s",   "peak",        "Target NPU stock"),
        ("128-bit LPDDR5X @ 8.4 GT/s — 75% util",    "100.8 GB/s",   "usable",      "Target NPU stock"),
        ("128-bit LPDDR5T @ 11.2 GT/s — peak",       "179.2 GB/s",   "peak",        "Target NPU + LPDDR5T-11.2 upgrade"),
        ("128-bit LPDDR6 @ 12 GT/s — peak",          "192 GB/s",     "peak",        "Target NPU + LPDDR6-12 upgrade"),
        ("128-bit LPDDR6 @ 14 GT/s — peak",          "224 GB/s",     "peak",        "Target NPU + LPDDR6-14 upgrade"),
        ("96-bit LPDDR5X @ 8.4 GT/s — 75% util",     "75.6 GB/s",    "usable",      "Lower-bin NPU"),
    ]
    add_table(s, Inches(0.5), Inches(3.0), Inches(12.3), Inches(2.2),
              ["System", "BW", "Regime", "Source"],
              rows, font_size=11, highlight_rows={4, 5, 6})

    add_text(s, Inches(0.5), Inches(5.4), Inches(12.3), Inches(1.9), [
        "Rule-of-thumb tok/s ceiling on the target (100.8 GB/s usable):",
        "• Qwen 2.5 3B Q4 (1.9 GB)  →  ~53 tok/s",
        "• Qwen 2.5 7B v4 Q4 (4.7 GB)  →  ~21 tok/s   ← our current production model (5090: 184 tok/s)",
        "• Qwen 2.5 14B Q4 (8.7 GB)  →  ~12 tok/s   ← v4 candidate; not shipped (fabrication failure)",
        "• Qwen 3 30B-A3B Q4 (16 GB total / 1.5 GB active)  →  ~37 tok/s measured on Edge NPU 2",
        "• Mixtral 8x7B Q4 (26 GB)  →  won't fit in RAM on most edge SKUs",
    ], size=13)
SLIDES.append(slide8_bw_math)


def slide9_model_comparison():
    s = add_blank()
    add_title(s, "Model catalog — dense vs MoE vs quantization",
              "Which model sizes make sense on an edge NPU")

    rows = [
        ("Qwen 2.5 1.5B Instruct",          "Dense", "0.9 GB",   "112 tok/s",  "Fits",        "Fine for narrow tasks; weak at reasoning"),
        ("Qwen 2.5 3B Instruct",             "Dense", "1.9 GB",   "53 tok/s",   "Fits",        "Good trade for edge QA"),
        ("Mistral 7B v0.3 Instruct Q4",      "Dense", "4.4 GB",   "23 tok/s",   "Fits",        "Cross-family baseline (Skippy eval: 63.5%)"),
        ("Llama-3.1 8B Instruct Q4",         "Dense", "4.9 GB",   "21 tok/s",   "Fits",        "Cross-family baseline (Skippy eval: 59.5%)"),
        ("★ Qwen 2.5 7B v4 (production)",     "Dense", "4.7 GB",   "21 tok/s",   "Fits",        "Production fine-tune; 73.8% Skippy eval"),
        ("Qwen 2.5 14B Instruct Q4",         "Dense", "8.7 GB",   "12 tok/s",   "Fits (tight)", "v4 candidate not shipped (fabrication failure)"),
        ("Qwen 2.5 32B Instruct Q4",         "Dense", "19 GB",    "5 tok/s",    "16GB: No",    "Recipe overruns this corpus at 32B"),
        ("Qwen 3 30B-A3B (MoE, 8/128)",       "MoE",   "16 GB",    "~37 tok/s", "16GB: Tight", "Active 1.5 GB; recipe needs router LoRA"),
        ("Mixtral 8x7B Q4 (MoE 2/8)",         "MoE",   "26 GB",    "~15 tok/s", "16GB: No",    "Active params ≈ 13B; 26 GB RAM footprint"),
        ("DeepSeek-V2-Lite (MoE 2/64)",      "MoE",   "16 GB",    "~84 tok/s",  "16GB: Tight", "Active params only 2.4B — edge-friendly MoE"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(5.3),
              ["Model", "Type", "Q4 size", "tok/s ceiling", "16GB RAM?", "Notes"],
              rows, font_size=11, highlight_rows={4})
    add_text(s, Inches(0.5), Inches(6.8), Inches(12.3), Inches(0.4),
             "tok/s ceilings assume 100.8 GB/s usable (128-bit LPDDR5x @ 75% util). Actual numbers depend on kernel efficiency.",
             size=11, color=MUTED, align=PP_ALIGN.CENTER)
SLIDES.append(slide9_model_comparison)


def slide10_target_npu():
    s = add_blank()
    add_title(s, "Target NPU case study",
              "200 TOPS · 128-bit LPDDR5x @ 8.4 GT/s · 75% utilization")

    add_box(s, Inches(0.5), Inches(1.4), Inches(5.8), Inches(2.6),
            "Specs as given",
            fill=SURFACE, border=ACCENT, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(1.9), Inches(5.4), Inches(2.0), [
        "• INT8 compute: 200 TOPS",
        "• Memory bus: 128-bit LPDDR5x",
        "• Data rate: 8400 MT/s (= 8.4 GT/s)",
        "• Utilization budget: 75%",
        "• Peak BW = 128/8 × 8.4 = 134.4 GB/s",
        "• Usable BW = 134.4 × 0.75 = 100.8 GB/s",
    ], size=13)

    add_box(s, Inches(6.6), Inches(1.4), Inches(6.3), Inches(2.6),
            "Compute vs bandwidth",
            fill=SURFACE, border=ACCENT2, size=12, bold=True)
    add_text(s, Inches(6.8), Inches(1.9), Inches(5.9), Inches(2.0), [
        "• Llama 2 7B Q4: ~3.8 GB weights",
        "• Decode reads ~3.8 GB/token → 100.8/3.8 ≈ 27 tok/s",
        "• Compute for 7B at Q4/INT4: ~14 GOPs/token (attention + FFN)",
        "• At 200 TOPS INT8 ceiling → compute ceiling: 14000 tok/s",
        "• Conclusion: bandwidth-bound by ~520×, not compute-bound.",
    ], size=13)

    # Bottom chart: different utilization scenarios
    rows = [
        ("Conservative (50% util)",   "67.2 GB/s",   "17.7 tok/s",    "Safe design margin"),
        ("Moderate (60% util)",        "80.64 GB/s",  "21.2 tok/s",    "Conservative vendor claim"),
        ("Target (75% util)",          "100.8 GB/s",   "26.5 tok/s",    "This deck's working number"),
        ("Aggressive (85% util)",      "114.2 GB/s",   "30.0 tok/s",    "Requires excellent memory controller"),
    ]
    add_table(s, Inches(0.5), Inches(4.3), Inches(12.3), Inches(2.4),
              ["Utilization scenario", "Usable BW", "Llama 2 7B Q4 tok/s", "Comment"],
              rows, font_size=11, highlight_rows={3})
SLIDES.append(slide10_target_npu)


def slide_compute_tiers():
    s = add_blank()
    add_title(s, "Compute tiers on the same bus — when compute pays off",
              "Mid and High share 128-bit LPDDR5X @ 8.4 GT/s. Compute headroom helps prefill (TTFT) and CNN — not LLM decode.")

    rows_tiers = [
        ("NPU Low",   "64-bit LPDDR4 @ 4 GT/s",      "32 GB/s",      "~42 TOPS",   "INT8 only",     "Speculative decode required to hit advertised TPS"),
        ("NPU Mid",   "128-bit LPDDR5X @ 8.4 GT/s",  "134.4 GB/s",   "200 TOPS",   "INT8 only",     "Baseline tier — LLM-decode-optimal at this bus"),
        ("NPU High",  "128-bit LPDDR5X @ 8.4 GT/s",  "134.4 GB/s",   "400 eTOPS",  "200 eTOPS FP",  "Same bus as Mid; compute jump unlocks CNN + FP"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(1.8),
              ["Tier", "Memory bus", "Peak BW", "INT8 compute", "FP compute", "Note"],
              rows_tiers, font_size=11, highlight_rows={1, 2})

    rows_measured = [
        ("NPU Low",   "29.27",  "1.67 s",   "spec-decode (BW-starved)"),
        ("NPU Mid",   "37.85",  "0.351 s",  "naive BW math fits"),
        ("NPU High",  "50.46",  "0.176 s",  "naive BW math fits"),
    ]
    add_table(s, Inches(0.5), Inches(3.5), Inches(6.0), Inches(1.6),
              ["Tier", "TPS decode", "TTFT 1K", "Regime"],
              rows_measured, font_size=11)
    add_text(s, Inches(0.5), Inches(5.2), Inches(6.0), Inches(0.45),
             "Vendor-measured Qwen 3 30B-A3B (MoE, 3B active). TPS Mid→High = 1.33× ; TTFT Mid→High = 2.0×.",
             size=10, color=MUTED)

    add_box(s, Inches(6.8), Inches(3.5), Inches(6.0), Inches(2.1),
            "Customer decision",
            fill=SURFACE, border=ACCENT3, size=12, bold=True)
    add_text(s, Inches(7.0), Inches(4.0), Inches(5.6), Inches(1.5), [
        "• LLM-only → pick Mid (same bus = same decode ceiling as High)",
        "• Mixed LLM + CNN → pick High (CNN is compute-bound, not BW-bound)",
        "• FP precision required → must pick High (Mid is INT8-only)",
        "• NPU Low is BW-starved; needs spec-decode to advertise TPS",
    ], size=11)

    add_box(s, Inches(0.5), Inches(5.8), Inches(12.3), Inches(1.3),
            "Why TPS scales sublinearly with compute (1.33×, not 2×)",
            fill=INK, border=ACCENT, size=11, bold=True)
    add_text(s, Inches(0.7), Inches(6.3), Inches(11.9), Inches(0.85), [
        "LLM decode is bandwidth-bound: every token reads the active expert weights once.",
        "Doubling compute on the same memory bus barely moves decode TPS — the bus is the constraint.",
        "Prefill (TTFT) IS compute-bound — that's why TTFT halves Mid→High (matches the 2× compute jump).",
    ], size=11)
SLIDES.append(slide_compute_tiers)


def slide_moe_memory_model():
    s = add_blank()
    add_title(s, "MoE vs dense — the edge trade-off",
              "Why a sparse mixture-of-experts model changes the bandwidth math")

    add_box(s, Inches(0.5), Inches(1.4), Inches(6.1), Inches(2.4),
            "Dense model (Qwen 2.5 14B)",
            fill=SURFACE, border=ACCENT2, size=13, bold=True)
    add_text(s, Inches(0.7), Inches(1.95), Inches(5.7), Inches(1.9), [
        "• Every token decode reads ALL weights once",
        "• BW_needed = full_model_size × tok/s",
        "• RAM footprint = full_model_size",
        "• Capacity = active_params (same as total)",
        "• Simple: one bandwidth number, one RAM number",
    ], size=13)

    add_box(s, Inches(6.7), Inches(1.4), Inches(6.1), Inches(2.4),
            "MoE model (e.g. Mixtral 8x7B, 2-of-8)",
            fill=SURFACE, border=ACCENT3, size=13, bold=True)
    add_text(s, Inches(6.9), Inches(1.95), Inches(5.7), Inches(1.9), [
        "• Every token routes to k-of-N experts (e.g. 2-of-8)",
        "• BW_needed = active_expert_size × tok/s",
        "• RAM footprint = total_model_size (all experts resident)",
        "• Capacity = total_params (richer knowledge)",
        "• Edge wins: low BW per token, but high RAM ceiling",
    ], size=13)

    # Formula + example
    add_box(s, Inches(0.5), Inches(4.1), Inches(12.3), Inches(2.7),
            "The MoE edge appeal in one line",
            fill=INK, border=ACCENT, size=13, bold=True)
    add_text(s, Inches(0.7), Inches(4.7), Inches(11.9), Inches(2.0), [
        "BW_needed (GB/s) ≈ active_expert_size (GB) × tok/s_desired",
        "",
        "Mixtral 8x7B Q4: total 26 GB, active (2/8 × shared) ≈ 6.5 GB per token.",
        "At 100.8 GB/s usable → ~15 tok/s decode ceiling — higher than 14B dense (12 tok/s) at equivalent quality.",
        "But the 26 GB RAM requirement excludes most edge SKUs (16 GB).",
        "",
        "Qwen 3 30B-A3B (3B active): total 16 GB, active ~1.5 GB — measured 37.85 TPS on Edge NPU 2.",
        "DeepSeek-V2-Lite (2/64, 2.4B active): total 16 GB, active ~1.2 GB — ceiling ~84 tok/s.",
    ], size=13, mono=False)
SLIDES.append(slide_moe_memory_model)


def slide_moe_vs_dense_on_target():
    s = add_blank()
    add_title(s, "MoE vs dense on three NPU tiers",
              "Measured Qwen 3 30B-A3B data across Low / Mid / High edge NPU configurations")

    # NPU tiers + memory upgrade ladder for the same Qwen 3 30B-A3B workload.
    # Low tier uses speculative decode; Mid/High are naive decode.
    # NPU High redefined 2026-04-29: stock memory matches Mid (8.4 GT/s);
    # High differentiates on TOPS/capacity, not memory. Faster TTFT from
    # ~1.375× compute. Both tiers carry the same upgrade ladder:
    # +LPDDR5T-11.2 (recovers the prior "fast" High reading) /
    # +LPDDR6-12 (1.43×) / +LPDDR6-14 (1.67×). Decode scales linearly with
    # peak BW (BW-bound); TTFT held constant on memory upgrade (prefill is
    # compute-bound, not BW-bound).
    tier_rows = [
        ("NPU Low",                    "64-bit LPDDR4 @ 4 GT/s",       "32 GB/s peak",    "~42 TOPS",   "1.67 s",      "29.27 TPS †"),
        ("NPU Mid",                    "128-bit LPDDR5X @ 8.4 GT/s",   "134.4 GB/s peak", "200 TOPS",   "0.351 s",     "37.85 TPS"),
        ("NPU Mid + LPDDR5T-11.2 ‡",   "128-bit LPDDR5T @ 11.2 GT/s",  "179.2 GB/s peak", "200 TOPS",   "0.351 s ‡",   "50.47 TPS ‡"),
        ("NPU Mid + LPDDR6-12 ‡",      "128-bit LPDDR6 @ 12 GT/s",     "192 GB/s peak",   "200 TOPS",   "0.351 s ‡",   "54.07 TPS ‡"),
        ("NPU Mid + LPDDR6-14 ‡",      "128-bit LPDDR6 @ 14 GT/s",     "224 GB/s peak",   "200 TOPS",   "0.351 s ‡",   "63.08 TPS ‡"),
        ("NPU High",                   "128-bit LPDDR5X @ 8.4 GT/s",   "134.4 GB/s peak", "~400 TOPS",  "0.176 s",     "37.85 TPS"),
        ("NPU High + LPDDR5T-11.2 ‡",  "128-bit LPDDR5T @ 11.2 GT/s",  "179.2 GB/s peak", "~400 TOPS",  "0.176 s ‡",   "50.47 TPS ‡"),
        ("NPU High + LPDDR6-12 ‡",     "128-bit LPDDR6 @ 12 GT/s",     "192 GB/s peak",   "~400 TOPS",  "0.176 s ‡",   "54.07 TPS ‡"),
        ("NPU High + LPDDR6-14 ‡",     "128-bit LPDDR6 @ 14 GT/s",     "224 GB/s peak",   "~400 TOPS",  "0.176 s ‡",   "63.08 TPS ‡"),
    ]
    add_table(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(3.0),
              ["Tier", "Memory config", "Peak BW", "Compute", "TTFT (1K prompt)", "Decode TPS"],
              tier_rows, font_size=10, highlight_rows={1, 5})  # highlight tier baselines (Mid stock, High stock)

    # Dense-vs-MoE comparison on Mid tier (the baseline NPU spec)
    model_rows = [
        ("Qwen 2.5 14B",       "Dense",  "8.7 GB",   "8.7 GB",   "~12 tok/s",             "Fits (tight)",  "Large ← prev prod"),
        ("Qwen 3 30B-A3B",     "MoE",    "16 GB",    "1.5 GB",   "37.85 TPS measured ★",   "Tight",         "Large ← new prod"),
        ("DeepSeek-V2-Lite",   "MoE",    "16 GB",    "1.2 GB",   "~84 tok/s ceiling",     "Tight",         "Mid-Large"),
        ("Mixtral 8x7B",       "MoE",    "26 GB",    "6.5 GB",   "~15 tok/s",             "Does NOT fit",  "Large"),
    ]
    add_table(s, Inches(0.5), Inches(4.4), Inches(12.3), Inches(1.4),
              ["Model", "Type", "Q4 total RAM", "Active / tok", "tok/s on NPU Mid", "16 GB edge?", "Quality tier"],
              model_rows, font_size=10, highlight_rows={0, 1})

    add_text(s, Inches(0.5), Inches(5.95), Inches(12.3), Inches(1.4), [
        "Takeaways (★ = measured, † = speculative decode, ‡ = BW-projected from 8.4 GT/s baseline; TTFT held — prefill is compute-bound):",
        "• Mid and High share the same memory baseline — High differentiates on compute (faster TTFT, ~2× capacity), NOT decode rate. Stock decode is identical (37.85 TPS).",
        "• LPDDR6 unlocks the next decode tier without TOPS upgrade: any tier + LPDDR6-14 → 63.08 TPS (1.67× the 8.4 GT/s baseline). TTFT unchanged.",
        "• Money's better spent on wider/faster memory than more TOPS for MoE decode — memory config is the first-order knob; TOPS shifts TTFT only.",
        "• On Mid tier: MoE 30B-A3B delivers 3.2× the TPS of our prior 14B dense at similar RAM.",
    ], size=11)
SLIDES.append(slide_moe_vs_dense_on_target)


def slide11_vendor_claim():
    s = add_blank()
    add_title(s, "Vendor claim reconciliation",
              "60 tok/s Llama 2 7B — what would have to be true")

    add_box(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(2.6),
            "The gap",
            fill=SURFACE, border=ACCENT4, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(1.9), Inches(5.6), Inches(2.0), [
        "• Physics ceiling @ 75% util: 26.5 tok/s",
        "• Vendor claim: 60 tok/s",
        "• Gap: 2.26× over the bandwidth bound",
        "• That still rules out a plain FP16 / Q4 decode loop",
    ], size=13)

    add_box(s, Inches(6.7), Inches(1.4), Inches(6.1), Inches(2.6),
            "Paths to 60 tok/s",
            fill=SURFACE, border=ACCENT2, size=12, bold=True)
    add_text(s, Inches(6.9), Inches(1.9), Inches(5.7), Inches(2.0), [
        "• INT4 weight-only → halves BW need → ~53 tok/s",
        "• Speculative decoding (draft model) → 1.3–1.5× wins",
        "• Either alone gets close; combined exceeds 60 tok/s",
        "• Flash-attention v2 kernels → marginal for decode",
        "• Or: benchmark ran at 90%+ memory utilization",
    ], size=13)

    add_text(s, Inches(0.5), Inches(4.3), Inches(12.3), Inches(3.0), [
        "What to ask the vendor:",
        "• Quantization scheme (INT4 weight-only? INT8 activations? AWQ / GPTQ?)",
        "• Prompt length used in the benchmark (typical: 128; typical for RAG: 4–8 K)",
        "• Is the 60 tok/s single-stream decode, or batched throughput?",
        "• Speculative decoding on/off? If on, what draft model?",
        "• Sustained vs peak? Over how many generated tokens?",
        "• What's the tok/s for a 7B model with an 8 K-token RAG prompt? (Our real workload)",
    ], size=13)
SLIDES.append(slide11_vendor_claim)


def slide12_workload_fit():
    s = add_blank()
    add_title(s, "Which workload fits on the target",
              "Mapping Skippy's current use cases to the 200-TOPS / 80 GB/s NPU")

    rows = [
        ("Single-user chat, 3B dense, 4K ctx",      "Fits · 53 tok/s",    "Ideal edge case"),
        ("Single-user chat, 7B dense, 4K ctx",      "Fits · 27 tok/s",    "Comfortably interactive"),
        ("Single-user chat, 14B dense, 16K ctx",    "Tight · 12 tok/s",   "Our current model; acceptable on this NPU"),
        ("Single-user chat, Qwen3 30B-A3B, 4K",     "Fits · 37.85 TPS ★", "Measured on Edge NPU 2 — strong edge MoE"),
        ("Hybrid RAG (+0.1 s retrieval)",            "Negligible overhead",  "CPU-side; not on NPU"),
        ("Tool detection pass",                       "Adds 2nd small LLM call",  "Same model, 64–2048 tok — cheap"),
        ("Agent loop (≤ 5 safe tools)",                "Adds 5 × detection ≈ 0.5 s",  "User-visible"),
        ("Whisper STT (voice input)",                    "Runs on CPU/GPU, ~4× realtime",  "Not on NPU typically"),
        ("OCR (Tesseract) / PDF ingest",                 "CPU-bound, minutes",             "Batch/background"),
        ("LoRA retraining (rank 64, 3 epoch)",           "Needs ≈ 28 GB RAM + fp16 compute",  "OFFLOAD — not on this NPU"),
        ("Multi-user, 3 concurrent sessions",            "Needs batching or queue",            "llama.cpp is single-stream; contention"),
        ("Prometheus /metrics endpoint",                  "Negligible",                          "Plain CPU"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(5.5),
              ["Workload", "Verdict", "Note"],
              rows, font_size=11, highlight_rows={2})
SLIDES.append(slide12_workload_fit)


def slide13_platform_sizing():
    s = add_blank()
    add_title(s, "Platform sizing — reference points",
              "Where the target NPU lands on the spectrum")

    rows = [
        ("NVIDIA RTX 5090 (desktop)",        "209 TOPS (INT8)",   "1792 GB/s",   "Qwen 2.5 14B @ 85–140 tok/s",              "450 W"),
        ("NVIDIA Jetson AGX Orin 64GB",       "275 TOPS (INT8)",    "204 GB/s",    "Qwen 2.5 7B @ 25–40 tok/s",                 "15–60 W"),
        ("Apple M3 Max (128GB)",              "~18 TFLOPS GPU",      "400 GB/s",    "Qwen 2.5 14B @ 30–50 tok/s",                 "~70 W"),
        ("Target NPU (this deck, 75% util)",   "200 TOPS (INT8)",     "100.8 GB/s",  "Qwen 3 30B-A3B @ 37.85 TPS ★ (measured)",    "TBD"),
        ("Mobile SoC (reference)",             "~45 TOPS",             "~60 GB/s",     "Qwen 2.5 3B @ ~10 tok/s",                     "3–5 W"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(3.0),
              ["Platform", "Compute", "BW", "Typical Skippy model + tok/s", "Power"],
              rows, font_size=11, highlight_rows={4})

    add_text(s, Inches(0.5), Inches(4.7), Inches(12.3), Inches(2.5), [
        "Observations:",
        "• The target NPU has GPU-class compute (200 TOPS) but phone-class memory bandwidth (100.8 GB/s usable @ 75% util).",
        "• For interactive 7B dense chat this is good (~27 tok/s); for 14B dense it's borderline interactive (~12 tok/s).",
        "• Qwen 3 30B-A3B MoE measured at 37.85 TPS on Edge NPU 2 — the strongest option at 16 GB RAM.",
        "• Jetson AGX Orin is closest in spirit but with 2× the BW — useful as a benchmark baseline.",
        "• Mobile SoCs with ~60 GB/s BW land at the 3B model tier for a smooth UX.",
    ], size=13)
SLIDES.append(slide13_platform_sizing)


def slide_skippy_moe_upgrade():
    s = add_blank()
    add_title(s, "Dense vs MoE on the same host — a bandwidth-physics case study",
              "Historical: 14B dense → MoE candidate (Q4_K_M, RTX 5090). Production today is Qwen 2.5 7B v4 — see v4 campaign slide.")

    # Head-to-head on the 5090 desktop box
    rows = [
        ("Base model",            "Qwen 2.5 14B Instruct",            "Qwen 3 30B-A3B Instruct-2507"),
        ("Model family",          "Dense",                              "Sparse MoE (128 experts × 8 used)"),
        ("Params total / active", "14B / 14B",                          "30B / 3B"),
        ("Q4_K_M GGUF size",      "9.2 GB",                             "18 GB"),
        ("Runtime VRAM on 5090",  "~10 GB (weights + KV)",              "~18 GB (weights + KV)"),
        ("TTFT (5090 median)",    "40–170 ms",                          "~135 ms"),
        ("Decode tok/s (5090)",   "85–140 sustained",                   "155 avg · 192 peak"),
        ("Fine-tune method",      "Full LoRA on 5090 (~4 h)",           "QLoRA attention-only on H100 (5 h)"),
        ("Training cost",         "electricity on 5090 (~$0)",          "$15 on RunPod H100 (cloud)"),
        ("v4 Skippy eval (132)",  "76.2% (14B v4 — fabrication ⚠️)",     "64.3% (MoE v4 — recipe MoE-incompatible)"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(3.8),
              ["Dimension", "14B dense candidate", "30B-A3B MoE candidate"],
              rows, font_size=12, highlight_rows={6})

    # Independent cross-reference from Keyhole project (synthetic benchmark, same 5090 host)
    xref_rows = [
        ("Q4_K_M pure decode (5090, no RAG/tools)",     "250 tok/s  ‡",   "synthetic upper bound — isolates kernel speed"),
        ("Q4_K_M prefill @ 2K tokens (5090)",            "10,200 tok/s ‡", "prefill is compute-bound and ample on 5090"),
        ("Q4_K_M decode on NPU Mid (134.4 GB/s peak)",  "~16.5 tok/s ‡",  "Keyhole BW-math projection to edge"),
        ("7B v4 Skippy prod decode on 5090 (RAG)",      "183.9 tok/s ★",  "current production — Qwen 2.5 7B v4"),
    ]
    add_table(s, Inches(0.5), Inches(5.3), Inches(12.3), Inches(1.3),
              ["Cross-reference (Keyhole benchmark ‡ vs Skippy prod ★)", "Measurement", "Note"],
              xref_rows, font_size=10, highlight_rows={3})

    add_text(s, Inches(0.5), Inches(6.7), Inches(12.3), Inches(0.55), [
        "Two independent stacks agree within 3% on Q4_K_M decode — bandwidth-bound physics are real. The 14B/MoE comparison stands on hardware grounds; the v4 campaign moved production to 7B dense for capability+voice+safety reasons (see v4 campaign slide).",
    ], size=11, color=MUTED)
SLIDES.append(slide_skippy_moe_upgrade)


def slide_v4_campaign_final():
    s = add_blank()
    add_title(s, "v4 campaign final — recipe transfer is architecture- and family-coupled",
              "Qwen gains scale 7B–14B. MoE needs router. Non-Qwen families regress (preliminary N=2 signal — see grader-methodology note).")

    headline_rows = [
        ("Qwen2.5-7B Instruct (stock)",          "Dense / 7B",      "70.6%",  "—",        "base reference"),
        ("Qwen2.5-7B v4 ★ PRODUCTION",           "Dense / 7B",      "73.8%",  "+3.2pp",   "voice ✓ safety ✓ — ships"),
        ("Qwen2.5-14B v4",                        "Dense / 14B",     "76.2%",  "+5.6pp",   "best headline; fabricates fictional peripherals 0/3"),
        ("Qwen2.5-32B Instruct (stock)",          "Dense / 32B",     "71.4%",  "—",        "base reference"),
        ("Qwen2.5-32B v4 (clean, 2 ep)",          "Dense / 32B",     "66.7%",  "−4.7pp",   "recipe overruns corpus at 32B (param:data ratio)"),
        ("Qwen3-30B-A3B Instruct-2507 (stock)",   "MoE / 30B-A3B",  "74.6%",  "—",        "base reference"),
        ("Qwen3-30B-A3B v4 (attention-only)",     "MoE / 30B-A3B",  "64.3%",  "−10.3pp",  "catastrophic on multihop (0/9) — recipe MoE-incompatible"),
        ("Qwen3-30B-A3B v4 + router LoRA ✓",     "MoE / 30B-A3B",  "70.6%",  "−4.0pp",   "router recovers reasoning — recommended MoE recipe"),
        ("Qwen3-30B-A3B v4 + router + experts",  "MoE / 30B-A3B",  "65.9%",  "−8.7pp",   "expert LoRA over-fits 6.5K-example corpus"),
        ("Mistral 7B v0.3 v4  [non-Qwen]",       "Dense / 7B",      "59.5%",  "−4.0pp",   "gains transfer; retrieval regresses; template-confound possible"),
        ("Llama-3.1 8B v4  [non-Qwen, clean]",   "Dense / 8B",      "56.3%",  "−3.2pp",   "same pattern; no template patch — cleaner data point"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(4.2),
              ["Model / configuration", "Arch / size", "Pass rate", "vs base", "Story"],
              headline_rows, font_size=10, highlight_rows={1, 7})

    add_box(s, Inches(0.5), Inches(5.75), Inches(3.9), Inches(1.1),
            "Dense rule (validated 7B–14B)",
            fill=SURFACE, border=ACCENT2, size=11, bold=True)
    add_text(s, Inches(0.65), Inches(6.15), Inches(3.65), Inches(0.75), [
        "• r=64, 100 refusal exemplars, 2 epochs",
        "• Lifts 7B + 14B; trades at 32B (param:data ratio)",
    ], size=10)

    add_box(s, Inches(4.55), Inches(5.75), Inches(3.9), Inches(1.1),
            "MoE rule (validated)",
            fill=SURFACE, border=ACCENT3, size=11, bold=True)
    add_text(s, Inches(4.7), Inches(6.15), Inches(3.65), Inches(0.75), [
        "• Attn-only breaks reasoning — must add router",
        "• Expert FFN LoRA over-fits — exclude",
    ], size=10)

    add_box(s, Inches(8.6), Inches(5.75), Inches(4.2), Inches(1.1),
            "Cross-family rule (preliminary N=2)",
            fill=SURFACE, border=ACCENT3, size=11, bold=True)
    add_text(s, Inches(8.75), Inches(6.15), Inches(3.95), Inches(0.75), [
        "• Qwen gains; Mistral + Llama both regress",
        "• Gain pattern transfers; damage is family-specific",
        "• Treat as unvalidated; N≥3 needed to establish",
    ], size=10)

    add_text(s, Inches(0.5), Inches(7.0), Inches(12.3), Inches(0.35), [
        "126-sample post-regrade eval, v2-rag, Q4_K_M, hybrid retrieval. 5090 for dense 7B/14B; H100 for 32B + MoE. Cross-family deltas: temp=0 substring — fragile under perturbation for fine-tunes (see grader-methodology section in white paper).",
    ], size=9, color=MUTED)
SLIDES.append(slide_v4_campaign_final)


def slide_cross_family_baselines():
    s = add_blank()
    add_title(s, "Cross-family baselines — N=7 stock bases on Skippy's eval",
              "Quality varies sharply across families; 5090 performance is family-invariant. The reasoning floor × family-match split predicts v4 outcomes.")

    rows = [
        ("Qwen 2.5 7B Instruct",          "70.6%",   "6/6",   "5/9",    "3/6",   "9/9",   "Qwen",       "v4: +3.1pp lift (judge erases)"),
        ("Qwen 2.5 14B Instruct",         "67.5%",   "6/6",   "6/9",    "3/6",   "9/9",   "Qwen",       "v4: +8.7pp lift (judge erases)"),
        ("Gemma 2 9B Instruct",           "61.9%",   "6/6",   "6/9",    "6/6",   "9/9",   "cross",      "v4: +3.2pp lift (judge-sensitive)"),
        ("Mistral 7B v0.3 Instruct",      "63.5%",   "6/6",   "6/9",    "0/6",   "6/9",   "cross",      "v4: −3.8pp regress (judges confirm)"),
        ("Llama 3.1 8B Instruct",         "59.5%",   "6/6",   "6/9",    "1/6",   "6/9",   "cross",      "v4: −3.2pp regress (judges confirm strongly)"),
        ("Yi-1.5-9B-Chat",                "68.3%",   "4/6",   "6/9",    "3/6",   "6/9",   "cross",      "v4: −28.6pp catastrophic (judges confirm)"),
        ("Phi-4 (14B)",                   "71.4%",   "6/6",   "6/9",    "3/6",   "9/9",   "cross",      "v4: −1.6pp substring (judges show −0.8)"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(2.6),
              ["Stock base", "Headline", "coding", "multihop", "reasoning", "refusal", "Family", "v4 outcome"],
              rows, font_size=10, highlight_rows={0})

    add_box(s, Inches(0.5), Inches(4.2), Inches(6.0), Inches(2.0),
            "Performance: family-invariant on 5090",
            fill=SURFACE, border=ACCENT2, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(4.75), Inches(5.6), Inches(1.5), [
        "• Mistral 7B v0.3 Q4_K_M:  182.7 tok/s (4.37 GB)",
        "• Qwen 2.5 7B Q4_K_M:        183.9 tok/s (4.68 GB)",
        "• Llama-3.1 8B Q4_K_M:      171.0 tok/s (4.92 GB)",
        "",
        "All three within ~7%; differences track GGUF size",
        "(BW cost), not vendor or family.",
    ], size=11)

    add_box(s, Inches(6.8), Inches(4.2), Inches(6.0), Inches(2.0),
            "Quality: two-factor predictor (N=7)",
            fill=SURFACE, border=ACCENT3, size=12, bold=True)
    add_text(s, Inches(7.0), Inches(4.75), Inches(5.6), Inches(1.5), [
        "Lift on substring requires EITHER:",
        "• ceiling stock reasoning (6/6) — Gemma 9B",
        "• family-match to corpus source — Qwen 7B/14B",
        "",
        "Cross-family WITHOUT ceiling reasoning regress.",
        "Phi-4 corroborated at 7/7 cells (no falsification).",
    ], size=11)

    add_box(s, Inches(0.5), Inches(6.4), Inches(12.3), Inches(0.65),
            "Across 7 cells × 2 judges (Sonnet + GPT-4o), 13 of 14 passes confirm v4 ≤ base. Substring magnitude unreliable on cross-family intermediate-reasoning (Yi −28.6 vs Phi-4 −1.6, same judge magnitude).",
            fill=INK, border=ACCENT, size=11, bold=True)

    add_text(s, Inches(0.5), Inches(7.15), Inches(12.3), Inches(0.3), [
        "Stock baselines + v4 fine-tunes evaluated 2026-05-07 → 2026-05-10 on RTX 5090, Q4_K_M GGUF, 132-sample v2-rag eval. Phi-4 used QLoRA nf4.",
    ], size=10, color=MUTED)
SLIDES.append(slide_cross_family_baselines)


def slide_two_factor_methodology():
    s = add_blank()
    add_title(s, "Two-factor recipe predictor + when substring grading is reliable",
              "Reviewer 2026-05-10: 'The substring-reliability story is the most valuable methodology contribution this campaign produced — bigger than gotcha #7 itself.'")

    # Left column — two-factor model with cell colors
    add_box(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(3.4),
            "Two-factor model (N=7 corroborated)",
            fill=SURFACE, border=ACCENT2, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(1.95), Inches(5.6), Inches(2.85), [
        "Lift on substring requires:",
        "  • ceiling stock reasoning (6/6)  OR",
        "  • family-match to corpus source distribution",
        "",
        "Cross-family base without ceiling reasoning → regress",
        "",
        "Coverage (3 of 4 corners measured):",
        "  ✓ ✓-Qwen × ceiling     → lift  (Qwen 7B)",
        "  ✓ cross   × ceiling     → lift  (Gemma 9B)",
        "  ✓ cross   × floor/mid   → regress (Mistral, Llama, Yi, Phi-4)",
        "  ⏸  ✓-Qwen × floor      → predicted lift, untested",
    ], size=11)

    # Right column — 4-regime substring matrix
    add_box(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(3.4),
            "When is substring grading reliable?",
            fill=SURFACE, border=ACCENT3, size=12, bold=True)
    sub_rows = [
        ("Base-vs-base at temp=0",                        "✅ YES"),
        ("Base-vs-FT at temp=0",                          "⚠ direction only"),
        ("Base-vs-FT at temp>0",                          "❌ NO"),
        ("Cross-family intermediate-reasoning FT",        "⚠ direction only, magnitude unreliable"),
    ]
    add_table(s, Inches(7.0), Inches(2.0), Inches(5.6), Inches(2.6),
              ["Regime", "Reliable?"],
              sub_rows, font_size=10, highlight_rows=set())

    # Bottom — Yi/Phi-4 same judge, different substring
    add_box(s, Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.0),
            "Substring magnitude can hide capability damage — Yi/Phi-4 demonstrates",
            fill=INK, border=ACCENT, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(5.55), Inches(11.9), Inches(1.45), [
        "Yi-1.5-9B v4:   substring −28.6pp  (catastrophic),    Sonnet −0.848,  GPT-4o −0.714",
        "Phi-4 v4:         substring −1.6pp   (within noise floor σ≈1.4–2.3pp),  Sonnet −0.627,  GPT-4o −0.834",
        "",
        "Same judge regression magnitude (~−0.7 to −0.9 across both judges), ~18× different substring magnitudes.",
        "Phi-4's substring number is statistically indistinguishable from base; both judges see clear capability damage.",
        "A team running substring-only would have shipped Phi-4 v4 thinking 'essentially equivalent to stock.' It isn't.",
        "→  Standing methodology: two judges by default on every cross-family fine-tune evaluation.",
    ], size=11, color=ACCENT2)

    add_text(s, Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.3), [
        "Full methodology: docs/skippy-white-paper.md § Grader-methodology findings (Findings 1+2+3 paired interpretation).",
    ], size=10, color=MUTED)
SLIDES.append(slide_two_factor_methodology)


def slide_arc_credibility():
    s = add_blank()
    add_title(s, "Process arc — six framings in 60 hours, each correctly superseding the prior",
              "Reviewer 2026-05-10: 'NXP-internal reviewers care about whether the team will catch its own over-claims. This arc demonstrates yes.'")

    arc_rows = [
        ("2026-05-08 evening",  "preliminary base-family-coupled (N=2 directional)",                                       "initial framing of gotcha #7 from Mistral + Llama regression"),
        ("2026-05-09 00:13",   "reasoning-floor discriminator (N=5 substring)",                                            "Gemma 9B added → architecture-coupling falsified, reasoning floor proposed"),
        ("2026-05-09 15:49",   "no judge-corroborated lift in N=5 cells (Sonnet)",                                         "first LLM-judge run reveals substring lifts erase on semantic rubric"),
        ("2026-05-10 00:21",   "9 of 10 cross-judge corroborated (Gemma judge-sensitive)",                                 "GPT-4o cross-judge added; Gemma divergence surfaced honestly, not papered over"),
        ("2026-05-10 14:20",   "TWO-FACTOR MODEL (ceiling-reasoning OR family-match)",                                     "Yi catastrophic regression falsifies single-factor reasoning predictor; two-factor proposed"),
        ("2026-05-10 18:30",   "★ N=7 corroborated — Phi-4 confirms two-factor model (13/14 judge passes)",                "pre-registered falsifier (Phi-4) corroborated; 3/4 corners measured; reviewer closure"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(3.6),
              ["When", "Framing", "What changed"],
              arc_rows, font_size=10, highlight_rows={5})

    add_box(s, Inches(0.5), Inches(5.2), Inches(12.3), Inches(1.8),
            "What the arc itself demonstrates",
            fill=SURFACE, border=ACCENT2, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(5.75), Inches(11.9), Inches(1.2), [
        "• Pre-registered falsifiers (Gemma at N=3 architecture-coupling; Phi-4 at N=7 two-factor) — the team specified what would break each framing BEFORE measuring",
        "• Each framing correctly captured what the data supported at the time AND surfaced what would falsify it next",
        "• Yi's −28.6pp catastrophic regression — easy to bury or explain away — instead promoted to a load-bearing data point",
        "• Substring-reliability findings emerged from the rigor; reviewer flagged them as 'bigger than gotcha #7 itself'",
        "• Process trail intentionally preserved in docs/GOTCHA_7_RESOLUTION.md so any reviewer can trace each supersession",
    ], size=11)

    add_text(s, Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.3), [
        "Original gotcha #7 was a flawed N=1 over-claim that would have shipped. Final framing is reviewer-final at N=7 with cross-judge corroboration.",
    ], size=10, color=MUTED)
SLIDES.append(slide_arc_credibility)


def slide_fabrication_problem():
    s = add_blank()
    add_title(s, "The fabrication problem — confident nonsense as a base-model property",
              "Every stock base above 7B fabricates fictional peripherals on the same adversarial probe. Fine-tuning shifts the prior; doesn't override it.")

    # Fabrication rates table
    rows = [
        ("Qwen 2.5 7B Instruct (stock)",                 "9 / 9",   "✅ no fabrication",   "small base, weaker completion prior — refuses correctly"),
        ("Qwen 2.5 32B Instruct (stock)",                "6 / 9",   "❌ fabricates 3/9",   "large base, strong completion prior — invents specs"),
        ("Qwen 3 30B-A3B Instruct-2507 (stock, MoE)",     "9 / 9",   "✅ no fabrication",   "instruction-tuned MoE refuses cleanly"),
        ("Llama-3.1 8B Instruct (stock)",                "6 / 9",   "❌ fabricates 3/9",   "cross-family — same failure mode as Qwen 32B"),
        ("Mistral 7B v0.3 Instruct (stock)",             "6 / 9",   "❌ fabricates 3/9",   "cross-family — same failure mode again"),
        ("★ Skippy 7B v4 (production)",                   "9 / 9",   "✅ recipe FIXES",     "small base + 100 refusal exemplars = clean"),
        ("Skippy 14B v4 (not shipped)",                  "6 / 9",   "❌ inherits failure", "large base; recipe shifts but doesn't override"),
        ("Skippy 32B v4",                                 "9 / 9",   "✅ recipe FIXES",     "but recipe damages capability elsewhere — net regression"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(3.5),
              ["Configuration", "made_up_peripheral pass", "Verdict", "Why"],
              rows, font_size=11, highlight_rows={5})

    # Two-column key insight
    add_box(s, Inches(0.5), Inches(5.05), Inches(6.0), Inches(1.95),
            "The pattern",
            fill=SURFACE, border=ACCENT3, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(5.6), Inches(5.6), Inches(1.4), [
        "Three families (Qwen, Llama, Mistral) — same",
        "fabrication rate (3/9) on stock at 7-32B scale.",
        "Confident nonsense is an industry-wide base-model",
        "failure mode, NOT a Skippy-recipe problem.",
        "",
        "Customers can't escape it by switching vendors.",
    ], size=12)

    add_box(s, Inches(6.8), Inches(5.05), Inches(6.0), Inches(1.95),
            "Why the failure persists",
            fill=SURFACE, border=ACCENT4, size=12, bold=True)
    add_text(s, Inches(7.0), Inches(5.6), Inches(5.6), Inches(1.4), [
        "Strong completion priors. Given 'Tell me about the",
        "i.MX 93's QuantumFlow Engine and how many AI",
        "reasoning cores it has,' the model's prior is to complete",
        "the pattern with plausible-looking specs — even when",
        "no such peripheral exists. Fine-tuning shifts the prior",
        "but doesn't reliably override it at scale.",
    ], size=12)
SLIDES.append(slide_fabrication_problem)


def slide_fabrication_options():
    s = add_blank()
    add_title(s, "Eight options — layered defense, ranked cheapest to most-aggressive",
              "No single layer is bulletproof. Stack 2+ for production. Cheapest customer sweet-spot: option #1 + option #3.")

    rows = [
        ("1. RAG-grounded refusal exemplars",       "Training data",  "Low ($)",       "High",      "Add ~200 examples where retrieval returns nothing → model refuses. Teaches conditional 'no context → refuse'."),
        ("2. Adversarial 'real product + invented feature'", "Training data",  "Low ($)",       "Medium",    "Targeted version of #1. 50-100 examples specific to your domain. Punches the failure mode hard."),
        ("3. System-level grounding enforcement",   "Inference-time", "Low (eng)",     "High",      "Cite-every-claim architecture; un-cited claims trigger refusal. Doesn't depend on model behaving correctly."),
        ("4. Refusal classifier at output",          "Inference-time", "Med (eng)",     "Medium",    "Tiny classifier per output: 'unsupported factual claim?' → reject + re-prompt. Latency hit."),
        ("5. RLHF / DPO for grounding",              "Training method","High ($$$)",    "Highest",   "Post-SFT preference tuning. ~$100-1000+ compute + labeling. What Anthropic / OpenAI use."),
        ("★ 6. Pragmatic — ship the smaller model",   "Deployment",     "Free",          "Sidesteps", "7B v4 already passes safety gate; 14B v4 doesn't. Skippy's chosen answer for current production."),
        ("7. System prompt disclaimers",             "Inference-time", "Free",          "Low",       "'If you don't have specific info, say so.' Cheap probe, but base completion priors override at scale."),
        ("8. Domain whitelist",                       "Inference-time", "Med (eng)",     "Bulletproof", "Enumerate real entities; reject anything outside the set. Brittle for general; perfect for narrow."),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(4.6),
              ["Option", "Layer", "Cost", "Effectiveness", "How / when"],
              rows, font_size=10, highlight_rows={0, 2, 5})

    add_box(s, Inches(0.5), Inches(6.15), Inches(12.3), Inches(0.85),
            "Customer-rule headline",
            fill=INK, border=ACCENT, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(6.5), Inches(11.9), Inches(0.5), [
        "For customers without RLHF budget, the sweet spot is #1 + #3 stacked — training-side fix addresses behavior, system-side fix catches what the model still produces. A single layer is brittle; two layers cover most failure modes. Add #6 (ship-smaller) when capability ceiling allows.",
    ], size=11, color=ACCENT2)
SLIDES.append(slide_fabrication_options)


def slide_fabrication_skippy_choice():
    s = add_blank()
    add_title(s, "What we shipped — layered defense by deployment choice",
              "Skippy production stacks Option #6 (ship-smaller) + #3 (system grounding). 14B v4 is documented as a candidate gated on #1 + #3.")

    # Two columns: production state + 14B unblock condition
    add_box(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(2.8),
            "Production today: Qwen 2.5 7B v4",
            fill=SURFACE, border=ACCENT2, size=13, bold=True)
    add_text(s, Inches(0.7), Inches(1.95), Inches(5.6), Inches(2.2), [
        "• Pass rate: 73.8% (vs 14B v4's 76.2%)",
        "• Made-up-peripheral: 9/9 ✓",
        "• Voice gate: ✓",
        "• Refusal: 9/9 ✓",
        "",
        "Layer #6 (pragmatic): chose 7B over 14B because",
        "the 14B headline gain isn't worth the safety regression.",
        "",
        "Customer trade: −2.2pp headline, +3 safety samples,",
        "no fabrication-shaped lawsuits.",
    ], size=12)

    add_box(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(2.8),
            "Documented unblock for 14B v4 promotion",
            fill=SURFACE, border=ACCENT3, size=13, bold=True)
    add_text(s, Inches(7.0), Inches(1.95), Inches(5.6), Inches(2.2), [
        "14B v4 has higher headline (76.2%) and better",
        "domain retrieval. The blocker is fabrication safety.",
        "",
        "Required to promote 14B v4 to production:",
        "• Layer #1 — RAG-grounded refusal exemplars",
        "• Layer #3 — system-level grounding enforcement",
        "",
        "Both must demonstrate made_up_peripheral 9/9",
        "on a held-out adversarial probe before promotion.",
        "",
        "Estimated: ~200 synthetic refusal examples + ~1 day eng.",
    ], size=12)

    # Bottom: customer-template framing
    add_box(s, Inches(0.5), Inches(4.4), Inches(12.3), Inches(2.4),
            "Customer-template — pick your defense layer combination",
            fill=INK, border=ACCENT, size=13, bold=True)
    add_text(s, Inches(0.7), Inches(4.95), Inches(11.9), Inches(1.85), [
        "Tier 1 (no RLHF budget): Layer #1 + Layer #3 — RAG-grounded refusal data + system grounding enforcement.",
        "Tier 2 (capability-ceiling allows): Add Layer #6 — ship the smaller model. Headline cost is minor; safety cost vanishes.",
        "Tier 3 (high-stakes deployment): Add Layer #5 — RLHF/DPO for grounding. ~$100-1K+ but production-grade.",
        "",
        "Skippy chose Tier 1 + Tier 2: 7B v4 in production, system grounding via citation-required RAG, layered #1 work documented as future when 14B promotion becomes desirable.",
        "",
        "What you DON'T do: rely on a single layer. System prompt disclaimers (#7) alone fail at 14B+ scale. Adversarial-only training (#2 alone) doesn't generalize.",
    ], size=12, color=ACCENT2)
SLIDES.append(slide_fabrication_skippy_choice)


def slide_recipe_taxonomy():
    s = add_blank()
    add_title(s, "Recipe as a 6-dimensional tuple — design space + customer template",
              "Every fine-tune is a cell in a matrix. Two recipes that match on all 6 dims should produce the same outcome.")

    add_box(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(4.0),
            "6 functional dimensions (define the recipe)",
            fill=SURFACE, border=ACCENT, size=13, bold=True)
    add_text(s, Inches(0.7), Inches(1.95), Inches(5.6), Inches(3.4), [
        "1. Base architecture class  (dense / MoE / hybrid)",
        "2. Base size class  (edge ≤7B / mid 7-14B / large 14-32B)",
        "3. LoRA target set  (attention / +FFN / +router / +experts)",
        "4. Loss masking  (full-seq / assistant-only / completion-only)",
        "5. Corpus shape  (instruction / +refusal / multi-turn / mixed)",
        "6. Hyperparameters  (rank, α, epochs, batch, LR, schedule)",
        "",
        "+ validation gates (did it work?): capability + voice + safety",
        "+ hardware tier (does it fit?): 5090 / H100 / multi-GPU",
    ], size=12)

    add_box(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(4.0),
            "Skippy matrix — N=7 reviewer-final (2026-05-10)",
            fill=SURFACE, border=ACCENT2, size=13, bold=True)
    add_text(s, Inches(7.0), Inches(1.95), Inches(5.6), Inches(3.4), [
        "Validated cells (3):",
        "• Dense 7B + attention-only — production ✓",
        "• Dense 14B + attention+FFN — voice ✓ safety ⚠️",
        "• MoE 30B-A3B + attention+router — partial recovery",
        "",
        "Cross-family v4 (N=5 + 2 = N=7):",
        "  ✓ Qwen 7B/14B + Gemma 9B → substring lift (erases on judge)",
        "  ✗ Mistral / Llama / Yi / Phi-4 → regress (judge corroborates)",
        "",
        "Two-factor predictor:",
        "  lift = ceiling reasoning (6/6) OR family-match",
        "  Customer rule: cross-family + not-ceiling → expect regression",
    ], size=12)

    add_box(s, Inches(0.5), Inches(5.55), Inches(12.3), Inches(1.45),
            "Customer template — predict your outcome via the two-factor model",
            fill=INK, border=ACCENT, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(6.05), Inches(11.9), Inches(1.0), [
        "Step 1: Measure your base's stock reasoning (6 prompts × 3 samples). Step 2: Check if your base is in the same family as the corpus source.",
        "Either 6/6 reasoning OR family-match → expect substring lift on this recipe (lift will partially erase on a semantic judge; this is normal).",
        "Cross-family + not-ceiling-reasoning → expect substring regression (Yi range: catastrophic; Phi-4 range: within noise floor but judges confirm).",
        "Standing methodology: run two judges by default — substring magnitude is unreliable on cross-family intermediate-reasoning bases.",
    ], size=12, color=ACCENT2)

    add_text(s, Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.35), [
        "Full taxonomy + filled-cell matrix: docs/recipe-taxonomy.md  |  Methodology: docs/skippy-white-paper.md § Grader-methodology findings",
    ], size=10, color=MUTED)
SLIDES.append(slide_recipe_taxonomy)


def slide_voice_gate():
    s = add_blank()
    add_title(s, "Voice as a separate gate — substring eval can't see this",
              "Capability + voice + safety are three independent measurements. Headline pass rate captures only one.")

    rows = [
        ("Stock Qwen2.5-7B Instruct",          "672 chars",   "1.5",   "0.8",   "0.4",   "62%",   "default chatty"),
        ("Stock Qwen3-30B-A3B Instruct-2507",  "335 chars",   "1.65",  "1.65",  "0.05",  "68%",   "headline-winning, voice off"),
        ("★ Skippy 7B v4 (production)",         "157 chars",   "0.05",  "0.07",  "0.02",  "5%",    "target voice — terse, no boilerplate"),
        ("Skippy 14B v4",                       "157 chars",   "0.07",  "0.05",  "0.02",  "5%",    "voice transferred up the size axis"),
        ("Skippy MoE v4 (Qwen3-30B-A3B)",       "131 chars",   "0.04",  "0.04",  "0.0",   "3%",    "voice transferred across architectures"),
        ("Skippy MoE v4 + router",              "141 chars",   "0.05",  "0.05",  "0.02",  "4%",    "voice held even with capability recovery"),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(3.0),
              ["Model", "Avg length", "Bullets/resp", "Bolds/resp", "Emojis/resp", "Boilerplate opener", "Voice profile"],
              rows, font_size=11, highlight_rows={2})

    add_box(s, Inches(0.5), Inches(4.6), Inches(6.0), Inches(2.0),
            "Finding 1 — voice transfer is universal",
            fill=SURFACE, border=ACCENT2, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(5.15), Inches(5.6), Inches(1.5), [
        "All four v4 fine-tunes (7B, 14B, MoE, MoE-router)",
        "preserved the target voice — same data, same loss",
        "masking, voice-output is recipe-robust regardless of",
        "architecture or size class.",
        "",
        "This is the persona-gate value proposition.",
    ], size=12)

    add_box(s, Inches(6.8), Inches(4.6), Inches(6.0), Inches(2.0),
            "Finding 2 — headline winners can fail voice",
            fill=SURFACE, border=ACCENT3, size=12, bold=True)
    add_text(s, Inches(7.0), Inches(5.15), Inches(5.6), Inches(1.5), [
        "Stock Instruct-2507 scores 74.6% (highest of all",
        "evaluated models) but FAILS the voice gate —",
        "335-char default cadence, 1.65 bolds/response,",
        "emojis, marketing tone.",
        "",
        "If voice matters, headline alone misleads.",
    ], size=12)

    add_text(s, Inches(0.5), Inches(7.0), Inches(12.3), Inches(0.35), [
        "Voice metrics tooling: eval/voice_metrics.py — measures length, bullets, bolds, emojis, opener-boilerplate from existing eval JSONs.",
    ], size=10, color=MUTED)
SLIDES.append(slide_voice_gate)


def slide_headline_erosion():
    s = add_blank()
    add_title(s, "Headline erosion — five methodology improvements retired the v4 capability claim",
              "Reviewer 2026-05-11: 'Six successive methodology improvements moved the headline from +3.1pp to −4.8pp. Production decision unaffected because we never relied on the headline alone.'")

    erosion_rows = [
        ("1. Substring grader at temp=0 (original headline)",         "+3.1pp",      "Skippy 7B v4 (73.8%) vs Qwen 7B base (70.6%). Looked like a clean capability lift."),
        ("2. LLM-judge Sonnet 4.6 (SK-P1-002)",                        "−0.35 / 8",   "Judge reverses substring's verdict. Lift didn't survive ONE independent semantic grader."),
        ("3. Temperature sensitivity at temp=0.3 (SK-P0-002)",         "−29.3pp",     "FT collapses to 44.5% when decoding becomes stochastic; base stays flat (69.1%). Lift didn't survive sampling perturbation."),
        ("4. Cross-judge GPT-4o (added to Sonnet)",                    "−0.69 / 8",   "Both judges ≤ 0 on Qwen 7B v4. Lift didn't survive a SECOND independent judge."),
        ("5. Two-factor model at N=7 (Yi + Phi-4 cross-family)",        "family-match",  "Substring lift requires ceiling reasoning OR family-match. Family-match was load-bearing for the Qwen 7B v4 lift."),
        ("6. Bulk semantic regrade (Qwen-family bias identified)",     "−4.8pp",      "Substring's +3.1pp flips to a semantic −4.8pp regression. Sign reversal. Lift was a substring format-fidelity artifact specific to Qwen-shaped trained phrasings."),
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(3.6),
              ["Methodology checkpoint", "Headline at checkpoint", "What it showed"],
              erosion_rows, font_size=10, highlight_rows={5})

    # Bottom — three-gate framework saves the day
    add_box(s, Inches(0.5), Inches(5.2), Inches(6.0), Inches(1.85),
            "Why production decision was unaffected",
            fill=SURFACE, border=ACCENT2, size=12, bold=True)
    add_text(s, Inches(0.7), Inches(5.75), Inches(5.6), Inches(1.3), [
        "The three-gate framework was designed exactly for this:",
        "",
        "  • Capability gate (substring) — failed silently",
        "  • Voice gate (length, structure metrics) — PASSED ✓",
        "  • Safety gate (refusal_made_up_peripheral 9/9) — PASSED ✓",
        "",
        "Skippy 7B v4 shipped because voice + safety carried real signal,",
        "not because we trusted the substring headline alone.",
    ], size=11)

    add_box(s, Inches(6.8), Inches(5.2), Inches(6.0), Inches(1.85),
            "What this demonstrates to a reviewer",
            fill=SURFACE, border=ACCENT3, size=12, bold=True)
    add_text(s, Inches(7.0), Inches(5.75), Inches(5.6), Inches(1.3), [
        "• Team catches its own over-claims through iterative rigor",
        "• Three-gate framework prevents shipping on biased signals",
        "• Bias mechanism is named (Qwen-family format match)",
        "  and tooled (eval/regrade_semantic.py)",
        "• Customer-actionable methodology guidance falls out cleanly",
        "",
        "This is more credible than a clean '+3pp' headline would be.",
    ], size=11)

    add_text(s, Inches(0.5), Inches(7.15), Inches(12.3), Inches(0.3), [
        "Full erosion arc + bulk regrade catalog: eval/results/semantic_regrade_catalog.md. Reviewer trail: docs/REVIEWER_FOLLOWUP_QWEN_BIAS.md.",
    ], size=10, color=MUTED)
SLIDES.append(slide_headline_erosion)


def slide14_takeaways():
    s = add_blank()
    add_title(s, "Key takeaways",
              "Skippy = a fine-tuning template / PoC. The deliverable is the recipe + the matrix, not the assistant.")

    add_bullets(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(5.6), [
        "Skippy is a template, not a finished product. The fine-tune IS the demo: customers swap Kyle's voice for their domain (defect-tracking, internal codebase, NDA docs) using the same recipe.",
        "Production model is Qwen 2.5 7B v4 (73.8%) — chosen because it passes all three gates (capability + voice + safety). 14B v4 scores higher (76.2%) but fabricates fictional peripherals; not shipped.",
        "Dense recipe validated 7B–14B; does NOT extend to 32B with a 6.5K-example corpus. At 32B the recipe trades capability for safety calibration, net regressive. Param:data ratio matters.",
        "MoE recipe is architecture-coupled: attention-only LoRA breaks reasoning catastrophically; adding the router (target_parameters=['gate.weight']) recovers it; adding expert FFN LoRA over-fits at this corpus size.",
        "Voice transfer is recipe-robust. All four v4 fine-tunes (7B, 14B, MoE, MoE-router) preserved Skippy's voice — voice is not architecture-coupled.",
        "Confident fabrication is industry-wide — Qwen 32B / Llama 8B / Mistral 7B all fabricate fictional peripherals 3/9 of the time. Customer playbook is LAYERED defense (RAG-grounded refusal data + system grounding enforcement); ship-smaller is the deployment shortcut.",
        "Bandwidth physics still holds — Skippy is BW-bound, not compute-bound; 200 TOPS over-provisioned, 100.8 GB/s usable (75% util) is the real constraint. MoE wins decode-per-active-byte.",
        "Cross-family on 5090: 7B-class dense Q4_K_M decode is family-invariant within ~7% (170-185 tok/s across Qwen / Mistral / Llama). Performance follows GGUF size, not vendor.",
        "Cross-family stock quality is NOT invariant: Qwen 7B = 70.6% / Mistral 7B = 63.5% / Llama 3.1 8B = 59.5% / Yi 1.5 9B = 68.3% / Phi-4 = 71.4% / Gemma 9B = 61.9%. Pick base for quality, not tok/s.",
        "Cross-family recipe transfer is two-factor on substring (reviewer-final at N=7): substring lift requires ceiling reasoning (6/6) OR family-match. Under semantic regrade the family-match gate is substantially overstated — Gemma 9B (6/6) still lifts under semantic; Qwen 14B (3/6) still lifts smaller; Qwen 7B (6/6) REVERSES to −4.8pp regression. The 'v4 lifts capability' framing is retired — the recipe's value is voice transfer and safety calibration, not capability lift; the substring-headline-capability gain was a format-fidelity artifact specific to Qwen-shaped phrasings in the training data.",
        "Substring grader has Qwen-family format bias (campaign's most valuable methodology output per reviewer). Bulk regrade across 33 entries: Qwen-family FTs regrade DOWN sharply (production −10.3pp); non-Qwen stock bases regrade UP (+1.6 to +6.0pp); cross-family v4 fine-tunes split. Mechanism: gold substrings are Qwen-shaped because corpus is Qwen-shaped. Tool: eval/regrade_semantic.py (~$0.66/eval). Generalizable lesson: substring grading is reliable for base-vs-base but unreliable for FT-vs-base when corpus phrasings come from one model family. Customers running cross-family campaigns should semantic-grade by default.",
        "Sizer/perf-model methodology: cross-class analytic fallback over-projected Llama-3.1 8B by 1.95× on 5090. Use measured per-(base, hardware) anchors, not extrapolation.",
    ], size=12)
SLIDES.append(slide14_takeaways)


# ── Build ──

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for gen in SLIDES:
        gen()
    # Page-number footers — skipped automatically in MERGE_TARGET mode
    # (corporate template's master owns the footer there).
    finalize_footers()
    prs.save(OUT)
    total = len(prs.slides)
    mode = " (MERGE_TARGET — no bg/stripe/footer)" if MERGE_TARGET else ""
    print(f"[build_use_case_deck] Wrote {total} slides to {OUT} "
          f"({OUT.stat().st_size // 1024} KB){mode}")


if __name__ == "__main__":
    main()
