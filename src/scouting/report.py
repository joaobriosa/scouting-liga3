"""Markdown report files written by more than one script.

reports/ranking_2025_26.md is written by rank_season.py (ranking sections) and
destinations_next.py (a trailing section). Each script replaces only its own
part, so the order in which they run does not matter.
"""

from __future__ import annotations

from pathlib import Path

DESTINATIONS_SECTION = "## Destinos 2026/27 (resultado parcial)"


def split_section(text: str, header: str) -> tuple[str, str]:
    """(text before the section, the section to the end of the file); the
    section part is "" if the header is absent."""
    i = text.find(header)
    return (text, "") if i < 0 else (text[:i], text[i:])


def write_main(path: Path, main_text: str, keep: str = DESTINATIONS_SECTION) -> None:
    """Write main_text, keeping the trailing `keep` section already in the file."""
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    _, tail = split_section(old, keep)
    out = main_text.rstrip() + "\n"
    if tail:
        out += "\n" + tail.rstrip() + "\n"
    path.write_text(out, encoding="utf-8")


def write_section(path: Path, section_text: str, header: str = DESTINATIONS_SECTION) -> None:
    """Replace (or append) the trailing section that starts with `header`."""
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    head, _ = split_section(old, header)
    out = (head.rstrip() + "\n\n" if head.strip() else "") + section_text.rstrip() + "\n"
    path.write_text(out, encoding="utf-8")
