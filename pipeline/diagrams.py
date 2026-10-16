"""Renders a small, fixed vocabulary of concept diagrams for video scenes whose
topic has no real filmable stock-footage equivalent (number patterns,
sequences, shape progressions, comparisons, trends) — e.g. dots forming a
triangular number, a number line with a highlighted point, or a company's
revenue trending across years.

Pure Pillow, no new dependency (textcard.py already uses it). Diagram specs
are validated against the same source text the LLM was given, mirroring
segmented_script._is_grounded's no-fabrication check for narration numbers,
so an on-screen diagram can never show a number the source material didn't.
"""
import re
from pathlib import Path

from PIL import Image, ImageDraw

from .textcard import _load_font, sanitize_display_text

BG_COLOR = (16, 22, 36)
ACCENT_COLOR = (60, 230, 110)
TEXT_COLOR = (230, 230, 230)

# Distinct per-slice colors for donut_share -- a single ACCENT_COLOR fill
# would make every slice indistinguishable from its neighbours.
DONUT_PALETTE = [
    (60, 230, 110), (240, 180, 40), (90, 160, 250), (235, 90, 90), (190, 120, 240),
]

# textcard.render_text_card's caption bar starts at 0.68*height (see
# pipeline/textcard.py) and is overlaid on top of a scene's clip -- including
# diagram scenes, since a diagram is the scene's entire visual (see
# producer.py/assemble.py). Confining all diagram content to this box keeps a
# diagram's own baked-in labels from landing underneath/overlapping that
# caption text.
CONTENT_HEIGHT_FRACTION = 0.62

DIAGRAM_KINDS = {
    "number_line": {"required": {"min": int, "max": int}, "optional": {"highlight": list}},
    "triangular_dots": {"required": {"rows": int}, "optional": {"label": str}},
    "sequence_boxes": {"required": {"items": list}, "optional": {"highlight_index": (int, type(None))}},
    "bar_compare": {"required": {"labels": list, "values": list}, "optional": {"unit": str}},
    "shape_progression": {"required": {"shape": str, "counts": list}, "optional": {}},
    # A connected line/curve rather than bar_compare's discrete bars -- suited
    # to a trend across an ordered sequence (years, quarters) rather than a
    # one-off comparison between a handful of unordered categories, e.g.
    # compound-interest growth over years or a company's revenue by year.
    "trend_line": {"required": {"labels": list, "values": list}, "optional": {"unit": str}},
    # A share-of-whole breakdown (e.g. revenue by segment, holding composition)
    # -- distinct from bar_compare, which compares magnitudes rather than parts
    # that sum to a whole.
    "donut_share": {"required": {"labels": list, "values": list}, "optional": {"unit": str}},
}

_DIGIT_RE = re.compile(r"\d{2,}")


def _numeric_labels(spec: dict) -> list:
    """All 2+-digit numeric tokens anywhere in the spec's values — single
    digits are too common (row counts, indices) to usefully ground."""
    out = []
    for v in spec.values():
        if isinstance(v, (int, float)):
            if len(re.sub(r"\D", "", str(v))) >= 2:
                out.append(str(v))
        elif isinstance(v, str):
            out += _DIGIT_RE.findall(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, (int, float)):
                    if len(re.sub(r"\D", "", str(item))) >= 2:
                        out.append(str(item))
                elif isinstance(item, str):
                    out += _DIGIT_RE.findall(item)
    return out


def validate_diagram_spec(spec: dict, source_text: str = "") -> tuple:
    """Returns (ok, reason). Rejects an unknown kind, a missing/mistyped
    required field, or any 2+-digit number that doesn't literally appear in
    source_text — a diagram must never show a fabricated figure."""
    if not isinstance(spec, dict) or "kind" not in spec:
        return False, "missing kind"
    kind = spec["kind"]
    if kind not in DIAGRAM_KINDS:
        return False, f"unknown kind {kind!r}"
    schema = DIAGRAM_KINDS[kind]
    for field, typ in schema["required"].items():
        if field not in spec or not isinstance(spec[field], typ):
            return False, f"missing/bad-typed required field {field!r}"
    for num in _numeric_labels(spec):
        # Source text often formats large numbers with thousands separators
        # (e.g. "₹1,112 crore") while a diagram's numeric value is a bare int
        # (1112) -- comparing against the comma-stripped source avoids
        # spuriously rejecting (and silently dropping) a correctly-grounded
        # diagram just because of punctuation.
        if num not in source_text.replace(",", ""):
            return False, f"ungrounded number {num!r}"
    return True, ""


def _fit_text(draw, text: str, max_width: int, start_size: int, min_size: int = 11):
    """Returns (font, text) sized/truncated so it renders within max_width.

    sequence_boxes/bar_compare were built assuming short items (numbers like
    "1", "10"), but an LLM-authored diagram can put a whole category name
    ("Triangular numbers") in the same slot -- at the fixed, box-width-derived
    font size used before this helper, that text drew far wider than its box
    and smeared into neighbouring boxes/bars. Shrinking the font first (down
    to min_size) and only then truncating with an ellipsis keeps every label
    inside its own slot instead of overlapping its neighbours."""
    text = sanitize_display_text(text)
    size = start_size
    while size > min_size:
        font = _load_font(size, text)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font, text
        size -= 2
    font = _load_font(min_size, text)
    truncated = text
    while truncated and draw.textbbox((0, 0), truncated + "…", font=font)[2] > max_width:
        truncated = truncated[:-1]
    return font, (truncated + "…" if truncated != text else text)


def render_diagram(spec: dict, width: int, height: int, out_path: Path) -> Path:
    kind = spec["kind"]
    renderer = {
        "number_line": _render_number_line,
        "triangular_dots": _render_triangular_dots,
        "sequence_boxes": _render_sequence_boxes,
        "bar_compare": _render_bar_compare,
        "shape_progression": _render_shape_progression,
        "trend_line": _render_trend_line,
        "donut_share": _render_donut_share,
    }.get(kind)
    if renderer is None:
        raise ValueError(f"Unknown diagram kind: {kind!r}")

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    # Confine the diagram itself to the band above the caption bar (see
    # CONTENT_HEIGHT_FRACTION) -- renderers lay their content out as if this
    # smaller height were the whole frame, which keeps it centered in the
    # free space above where on_screen_text gets overlaid afterwards.
    content_height = int(height * CONTENT_HEIGHT_FRACTION)
    renderer(draw, spec, width, content_height)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _render_number_line(draw, spec, width, height):
    lo, hi = spec["min"], spec["max"]
    highlight = set(spec.get("highlight") or [])
    y = height // 2
    x0, x1 = int(width * 0.08), int(width * 0.92)
    draw.line([(x0, y), (x1, y)], fill=TEXT_COLOR, width=4)
    n = max(1, hi - lo)
    font = _load_font(max(28, width // 22))
    for i in range(lo, hi + 1):
        x = x0 + int((x1 - x0) * (i - lo) / n)
        fill = ACCENT_COLOR if i in highlight else TEXT_COLOR
        draw.line([(x, y - 14), (x, y + 14)], fill=fill, width=4)
        draw.text((x, y + 24), str(i), font=font, fill=fill, anchor="ma")


def _render_triangular_dots(draw, spec, width, height):
    rows = spec["rows"]
    label = spec.get("label", "")
    radius = max(6, min(width, height) // (rows * 6 + 4))
    gap = radius * 3
    total_h = rows * gap
    top_y = height // 2 - total_h // 2
    for r in range(1, rows + 1):
        row_w = (r - 1) * gap
        start_x = width // 2 - row_w // 2
        y = top_y + (r - 1) * gap
        for c in range(r):
            x = start_x + c * gap
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=ACCENT_COLOR)
    if label:
        font_size = max(28, width // 20)
        font, label = _fit_text(draw, label, int(width * 0.92), font_size)
        label_y = min(top_y + rows * gap + radius * 2, height - font_size - 4)
        draw.text((width // 2, label_y), label, font=font, fill=TEXT_COLOR, anchor="ma")


def _render_sequence_boxes(draw, spec, width, height):
    items = [str(i) for i in spec["items"]][:8]
    hi_idx = spec.get("highlight_index")
    n = len(items)
    # box_w is sized so that n boxes PLUS the (n-1) gaps between them (each
    # gap = 0.3*box_w) fit within width*0.8 -- sizing box_w off box count
    # alone and adding gaps afterwards (the old formula) ignored the gaps'
    # own width, so e.g. n=8 overflowed the frame by ~10px on each side,
    # clipping the first/last box at the frame edge.
    box_w = min(int(width * 0.8 / max(1, n + 0.3 * (n - 1))), int(width * 0.18))
    box_h = box_w
    gap = int(box_w * 0.3)
    total_w = n * box_w + (n - 1) * gap
    x0 = (width - total_w) // 2
    y0 = height // 2 - box_h // 2
    for i, item in enumerate(items):
        x = x0 + i * (box_w + gap)
        fill = ACCENT_COLOR if i == hi_idx else (40, 50, 70)
        draw.rectangle([x, y0, x + box_w, y0 + box_h], fill=fill, outline=TEXT_COLOR, width=3)
        font, label = _fit_text(draw, item, int(box_w * 0.85), max(24, box_w // 3))
        draw.text((x + box_w // 2, y0 + box_h // 2), label, font=font, fill=(255, 255, 255), anchor="mm")


def _render_bar_compare(draw, spec, width, height):
    labels, values = spec["labels"][:6], spec["values"][:6]
    unit = (spec.get("unit") or "").strip()
    max_v = max(values) if values else 1
    n = len(values)
    bar_w = int(width * 0.7 / max(1, n))
    gap = int(bar_w * 0.4)
    total_w = n * bar_w + (n - 1) * gap
    x0 = (width - total_w) // 2
    base_y = int(height * 0.75)
    max_h = int(height * 0.5)
    for i, (label, v) in enumerate(zip(labels, values)):
        h = int(max_h * (v / max_v)) if max_v else 0
        x = x0 + i * (bar_w + gap)
        draw.rectangle([x, base_y - h, x + bar_w, base_y], fill=ACCENT_COLOR)
        label_font, label_text = _fit_text(draw, str(label), int(bar_w * 0.9), max(22, bar_w // 4))
        draw.text((x + bar_w // 2, base_y + 10), label_text, font=label_font, fill=TEXT_COLOR, anchor="ma")
        value_display = f"{v} {unit}".strip() if unit else str(v)
        value_font, value_text = _fit_text(draw, value_display, int(bar_w * 0.9), max(22, bar_w // 4))
        draw.text((x + bar_w // 2, base_y - h - 10), value_text, font=value_font, fill=(255, 255, 255), anchor="ms")


def _render_shape_progression(draw, spec, width, height):
    shape, counts = spec["shape"], spec["counts"][:5]
    n = len(counts)
    cell_w = width // max(1, n)
    font = _load_font(max(24, cell_w // 8))
    for i, count in enumerate(counts):
        cx = cell_w * i + cell_w // 2
        cy = height // 2
        r = max(8, min(cell_w, height) // 10)
        cols = max(1, int(count ** 0.5))
        for k in range(count):
            row, col = divmod(k, cols)
            x = cx - (cols * r) + col * 2 * r
            y = cy - r * 2 + row * 2 * r
            if shape == "square":
                draw.rectangle([x - r // 2, y - r // 2, x + r // 2, y + r // 2], fill=ACCENT_COLOR)
            else:
                draw.ellipse([x - r // 2, y - r // 2, x + r // 2, y + r // 2], fill=ACCENT_COLOR)
        draw.text((cx, cy + r * 3), str(count), font=font, fill=TEXT_COLOR, anchor="ma")


def _render_trend_line(draw, spec, width, height):
    labels, values = spec["labels"][:8], spec["values"][:8]
    unit = (spec.get("unit") or "").strip()
    n = len(values)
    max_v, min_v = max(values), min(0, min(values))
    span = max_v - min_v or 1
    x0, x1 = int(width * 0.08), int(width * 0.92)
    base_y = int(height * 0.72)
    top_y = int(height * 0.12)
    step_x = (x1 - x0) / max(1, n - 1)
    points = []
    for i, v in enumerate(values):
        x = x0 + int(step_x * i)
        y = base_y - int((base_y - top_y) * (v - min_v) / span)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=ACCENT_COLOR, width=5, joint="curve")
    label_font_size = max(20, width // (n * 10 + 10))
    for (x, y), label, v in zip(points, labels, values):
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=ACCENT_COLOR)
        value_display = f"{v} {unit}".strip() if unit else str(v)
        value_font, value_text = _fit_text(draw, value_display, int(step_x * 0.9) or width, max(22, label_font_size))
        draw.text((x, y - 16), value_text, font=value_font, fill=(255, 255, 255), anchor="ms")
        label_font, label_text = _fit_text(draw, str(label), int(step_x * 0.9) or width, label_font_size)
        draw.text((x, base_y + 14), label_text, font=label_font, fill=TEXT_COLOR, anchor="ma")


def _render_donut_share(draw, spec, width, height):
    labels, values = spec["labels"][:5], spec["values"][:5]
    unit = (spec.get("unit") or "").strip()
    total = sum(values) or 1
    cx, cy = int(width * 0.3), height // 2
    radius = min(cx, height // 2) - 10
    inner_radius = int(radius * 0.55)
    start_angle = -90.0
    colors = []
    for i, v in enumerate(values):
        sweep = 360.0 * v / total
        color = DONUT_PALETTE[i % len(DONUT_PALETTE)]
        colors.append(color)
        draw.pieslice(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            start=start_angle, end=start_angle + sweep, fill=color,
        )
        start_angle += sweep
    draw.ellipse([cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius], fill=BG_COLOR)

    legend_x = int(width * 0.56)
    row_h = max(28, int(height / (len(labels) * 1.6)))
    legend_top = cy - (len(labels) * row_h) // 2
    swatch = max(14, row_h // 3)
    text_x = legend_x + swatch + 10
    max_text_width = int(width * 0.96) - text_x
    for i, (label, v, color) in enumerate(zip(labels, values, colors)):
        y = legend_top + i * row_h
        draw.rectangle([legend_x, y, legend_x + swatch, y + swatch], fill=color)
        pct = round(100 * v / total)
        text = f"{label} {pct}%" if not unit else f"{label} {v}{unit}"
        font, text = _fit_text(draw, text, max_text_width, max(20, row_h // 2))
        draw.text((text_x, y + swatch // 2), text, font=font, fill=TEXT_COLOR, anchor="lm")

