import re
from dataclasses import dataclass


ATTRACTION_HEADER_RE = re.compile(r"^(?:\d+\.\s*)?###\s*(?:\d+\.\s*)?(.+)$")
ATTRACTION_ID_RE = re.compile(r"\s*\[id:\s*([^\]]+)\]\s*$")


@dataclass(frozen=True)
class AttractionSection:
    title: str
    content: str
    attraction_id: str | None = None


def parse_attraction_sections(markdown: str) -> list[AttractionSection]:
    sections: list[AttractionSection] = []
    current_title: str | None = None
    current_id: str | None = None
    current_lines: list[str] = []

    def finish_section() -> None:
        if current_title is not None:
            sections.append(
                AttractionSection(
                    title=current_title,
                    content="\n".join(current_lines).strip(),
                    attraction_id=current_id,
                )
            )

    for line in markdown.splitlines():
        header_match = ATTRACTION_HEADER_RE.match(line)
        if not header_match:
            if current_title is not None:
                current_lines.append(line)
            continue

        finish_section()
        title = header_match.group(1).strip()
        id_match = ATTRACTION_ID_RE.search(title)
        current_id = id_match.group(1).strip() if id_match else None
        current_title = title[:id_match.start()].rstrip() if id_match else title
        current_lines = []

    finish_section()
    return sections