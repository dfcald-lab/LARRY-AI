import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "alice_wonderland.txt"

START_MARKER = "START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "END OF THE PROJECT GUTENBERG EBOOK"

CHAPTER_PATTERN = re.compile(
    r"^\s*CHAPTER\s+([IVXLCDM]+)\.\s*$",
    re.IGNORECASE,
)

EXPECTED_CHAPTERS = [
    "I", "II", "III", "IV", "V", "VI",
    "VII", "VIII", "IX", "X", "XI", "XII",
]

EDITORIAL_START = "[later editions continued as follows"


def extract_chapters(book_text):
    lines = book_text.splitlines()

    start = next(
        (
            i for i, line in enumerate(lines)
            if START_MARKER in line
        ),
        None,
    )
    end = next(
        (
            i for i, line in enumerate(lines)
            if END_MARKER in line
        ),
        None,
    )

    if start is None or end is None or end <= start:
        raise ValueError("Could not locate Project Gutenberg book markers.")

    body = lines[start + 1:end]
    headings = []

    for index, line in enumerate(body):
        match = CHAPTER_PATTERN.fullmatch(line)
        if match:
            headings.append((index, match.group(1).upper()))

    found = [number for _, number in headings]

    if found != EXPECTED_CHAPTERS:
        raise ValueError(
            f"Expected 12 chapter headings; found {found!r}"
        )

    chapters = []

    for position, (heading_line, number) in enumerate(headings):
        if position + 1 < len(headings):
            next_heading = headings[position + 1][0]
        else:
            next_heading = len(body)

        # Skip the chapter heading and the following chapter title.
        content_start = heading_line + 2
        chapter_lines = body[content_start:next_heading]
        chapters.append((number, chapter_lines))

    return chapters


def clean_chapter_lines(lines):
    """Remove editorial insertions and join wrapped text paragraphs."""

    paragraphs = []
    current_lines = []
    inside_editorial = False

    def finish_paragraph():
        if current_lines:
            paragraphs.append(" ".join(current_lines))
            current_lines.clear()

    for line in lines:
        stripped = line.strip()

        if inside_editorial:
            if "]" in stripped:
                inside_editorial = False
            continue

        if stripped.lower().startswith(EDITORIAL_START):
            finish_paragraph()
            if "]" not in stripped:
                inside_editorial = True
            continue

        if not stripped:
            finish_paragraph()
            continue

        current_lines.append(stripped)

    finish_paragraph()

    if inside_editorial:
        raise ValueError("An editorial insertion did not close.")

    return "\n\n".join(paragraphs).strip()


def main():
    if not SOURCE.is_file():
        raise SystemExit(
            "Missing alice_wonderland.txt. Place the original "
            "Project Gutenberg download in this experiment folder."
        )

    book_text = SOURCE.read_text(encoding="utf-8")
    chapters = extract_chapters(book_text)

    train_parts = [
        clean_chapter_lines(lines)
        for number, lines in chapters[:8]
    ]
    held_out_parts = [
        clean_chapter_lines(lines)
        for number, lines in chapters[8:]
    ]

    train = "\n\n".join(part for part in train_parts if part)
    held_out = "\n\n".join(part for part in held_out_parts if part)

    if not train or not held_out:
        raise SystemExit("Training or held-out text is empty.")

    (ROOT / "train_clean.txt").write_text(
        train + "\n", encoding="utf-8"
    )
    (ROOT / "held_out_clean.txt").write_text(
        held_out + "\n", encoding="utf-8"
    )

    training_characters = set(train)
    unseen = Counter(
        character
        for character in held_out
        if character not in training_characters
    )

    print("EXPERIMENT 082 — DATA PREPARATION")
    print(f"Source file: {SOURCE.name}")
    print(f"Chapters extracted: {len(chapters)}")
    print("Training chapters: I–VIII")
    print("Held-out chapters: IX–XII")
    print(f"Training characters: {len(train)}")
    print(f"Held-out characters: {len(held_out)}")
    print(f"Training paragraphs: {len(train.split(chr(10) + chr(10)))}")
    print(f"Held-out paragraphs: {len(held_out.split(chr(10) + chr(10)))}")
    print(f"Unseen held-out characters: {len(unseen)}")

    for character, count in sorted(unseen.items()):
        print(
            f"  {character!r} (U+{ord(character):04X}): {count}"
        )

    if unseen:
        raise SystemExit(
            "Review unseen characters before running the model."
        )

    if CHAPTER_PATTERN.search(train) or CHAPTER_PATTERN.search(held_out):
        raise SystemExit("A chapter heading remains after cleaning.")

    if EDITORIAL_START in train.lower() or EDITORIAL_START in held_out.lower():
        raise SystemExit("An editorial insertion remains after cleaning.")

    print("PASS: both datasets are ready for evaluation.")


if __name__ == "__main__":
    main()
