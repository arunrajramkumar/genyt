#!/usr/bin/env python3
"""Turns a folder of NCERT-style textbook chapter PDFs into a themes_education.yaml
queue: one video per group of 2-ish consecutive numbered sections (so "parts"
follow the book's own chapter/section structure instead of being arbitrary cuts),
with the real textbook text embedded in the prompt as grounding so the LLM
explains the actual syllabus content rather than inventing its own curriculum.

Usage:
    python3 scripts/build_education_queue.py <pdf_folder> <output_yaml> \
        --subject "Grade 6 CBSE Maths (NCERT Ganita Prakash)" \
        --chapter-titles '{"1": "Patterns in Mathematics", "2": "Lines and Angles", ...}' \
        --base-tags "NCERT,CBSE,Grade 6,6th Grade,School Maths,India,Learning,Students" \
        --thumbnail-badge "CBSE | Grade 6 Maths"
"""
import argparse
import glob
import json
import re

import yaml

HEADER_RE = re.compile(r"^\s*(\d+)\.(\d+)\s+([A-Z][^\n]{2,80}?)\s*$", re.MULTILINE)
BOILERPLATE_RE = re.compile(r"^(Reprint \d{4}-\d{2}|Ganita Prakash \| Grade 6|\d{1,3})\s*$", re.MULTILINE)

MAX_SECTIONS_PER_PART = 2  # keeps each video to one sitting of focused concepts
MAX_PARTS_PER_CHAPTER = 6
MAX_SOURCE_CHARS_PER_PART = 8000  # keeps the grounding prompt within a sane LLM context budget


def extract_sections(pdf_path: str, chapter_num: int) -> list[dict]:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    headers = [m for m in HEADER_RE.finditer(text) if int(m.group(1)) == chapter_num]

    sections = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = BOILERPLATE_RE.sub("", text[start:end]).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        sections.append({"id": f"{m.group(1)}.{m.group(2)}", "title": m.group(3).strip(), "body": body})
    return sections


def group_sections(sections: list[dict]) -> list[list[dict]]:
    n = len(sections)
    if n == 0:
        return []
    num_parts = min(MAX_PARTS_PER_CHAPTER, max(1, -(-n // MAX_SECTIONS_PER_PART)))
    base, extra = divmod(n, num_parts)
    groups, i = [], 0
    for g in range(num_parts):
        size = base + (1 if g < extra else 0)
        groups.append(sections[i:i + size])
        i += size
    return [g for g in groups if g]


def build_theme_entry(subject: str, chapter_num: int, chapter_title: str,
                       part_num: int, total_parts: int, group: list[dict], base_tags: list[str],
                       thumbnail_badge: str = None) -> dict:
    section_titles = " & ".join(s["title"] for s in group)
    source_text = "\n\n".join(f'{s["id"]} {s["title"]}\n{s["body"]}' for s in group)[:MAX_SOURCE_CHARS_PER_PART]

    theme = (
        f'Explain these {subject} concepts from Chapter {chapter_num} "{chapter_title}": '
        f"{section_titles}.\n\n"
        "Use ONLY the mathematical facts, definitions, and rules described in the "
        "textbook excerpt below -- do not introduce incorrect or additional "
        "mathematical claims. Make it very easy to understand for an 11-12 year old: "
        "explain each idea simply, step by step, and illustrate every concept with "
        "your own original real-life example a school student would relate to.\n\n"
        f'Textbook excerpt:\n"""\n{source_text}\n"""'
    )

    duration_sec = min(360, max(180, 90 * len(group) + 90))
    return {
        "theme": theme,
        "style": f"clear, patient, example-rich teacher explaining {subject} to 11-12 year olds",
        "duration_sec": duration_sec,
        "shorts": False,
        "done": False,
        # Tags that must appear on every video regardless of what the LLM comes
        # up with (see segmented_script.merge_tags) -- the chapter title is
        # appended so each video is also tagged with its specific topic.
        "extra_tags": [*base_tags, f"Chapter {chapter_num}", chapter_title],
        # Corner badge stamped on the thumbnail/hook card (see textcard.render_hook_card)
        # so the syllabus/grade is visible even in a small thumbnail grid.
        "thumbnail_badge": thumbnail_badge,
        # Not read by run.py -- just makes the generated queue file human-readable.
        "_label": f"Ch {chapter_num} ({chapter_title}) - Part {part_num}/{total_parts}: {section_titles}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_folder")
    parser.add_argument("output_yaml")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--chapter-titles", required=True, help='JSON map of chapter number (as string) -> title')
    parser.add_argument("--pattern", default="*.pdf")
    parser.add_argument(
        "--base-tags", default="",
        help="Comma-separated tags added to every video's extra_tags, e.g. 'NCERT,CBSE,Grade 6,India'",
    )
    parser.add_argument(
        "--thumbnail-badge", default=None,
        help="Corner label stamped on every video's thumbnail, e.g. 'CBSE | Grade 6 Maths'",
    )
    args = parser.parse_args()

    base_tags = [t.strip() for t in args.base_tags.split(",") if t.strip()]
    chapter_titles = json.loads(args.chapter_titles)
    entries = []
    for pdf_path in sorted(glob.glob(f"{args.pdf_folder}/{args.pattern}")):
        match = re.search(r"(\d{2})(?=\.pdf$)", pdf_path)
        if not match:
            print(f"Skipping {pdf_path} (no chapter number found in filename)")
            continue
        chapter_num = int(match.group(1))
        if str(chapter_num) not in chapter_titles:
            print(f"Skipping {pdf_path} (no title mapping for chapter {chapter_num})")
            continue
        chapter_title = chapter_titles[str(chapter_num)]

        sections = extract_sections(pdf_path, chapter_num)
        groups = group_sections(sections)
        print(f"Chapter {chapter_num} ({chapter_title}): {len(sections)} sections -> {len(groups)} parts")
        for part_num, group in enumerate(groups, start=1):
            entries.append(build_theme_entry(
                args.subject, chapter_num, chapter_title, part_num, len(groups), group, base_tags,
                args.thumbnail_badge,
            ))

    with open(args.output_yaml, "w") as f:
        yaml.safe_dump({"themes": entries}, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"Wrote {len(entries)} theme entries to {args.output_yaml}")


if __name__ == "__main__":
    main()
