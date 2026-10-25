from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class ParsedLogEntry:
    line_number: int
    timestamp: str | None
    level: str | None
    message: str

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


_TIMESTAMP_PATTERNS = (
    re.compile(
        r"^\s*\[?(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
        r"(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\]?\s*"
    ),
    re.compile(
        r"^\s*\[?(?P<timestamp>\d{2,4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}"
        r"(?:[.,]\d+)?)\]?\s*"
    ),
    re.compile(
        r"^\s*\[?(?P<timestamp>\d{1,2}/[A-Za-z]{3}/\d{4}:"
        r"\d{2}:\d{2}:\d{2}(?:\s+[+-]\d{4})?)\]?\s*"
    ),
    re.compile(
        r"^\s*\[?(?P<timestamp>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\]?\s*",
        re.IGNORECASE,
    ),
)
_LEVEL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(EMERGENCY|CRITICAL|CRIT|FATAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE|NOTICE|ALERT|PANIC)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def parse_log_lines(log_contents: str) -> list[ParsedLogEntry]:
    """Extract timestamp, severity level, and message from each non-empty log line."""
    parsed_entries = []

    for line_number, raw_line in enumerate(log_contents.splitlines(), start=1):
        if not raw_line.strip():
            continue

        remainder = raw_line
        timestamp = None
        for pattern in _TIMESTAMP_PATTERNS:
            timestamp_match = pattern.match(raw_line)
            if timestamp_match is not None:
                timestamp = timestamp_match.group("timestamp")
                remainder = raw_line[timestamp_match.end() :]
                break

        level_match = _LEVEL_PATTERN.search(remainder)
        if level_match is None:
            level = None
            message = remainder.strip(" \t:|-]")
        else:
            level = level_match.group().upper()
            message = remainder[level_match.end() :].strip(" \t:|-]")

        parsed_entries.append(
            ParsedLogEntry(
                line_number=line_number,
                timestamp=timestamp,
                level=level,
                message=message,
            )
        )

    return parsed_entries
