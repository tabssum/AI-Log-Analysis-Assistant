from dataclasses import dataclass

from log_parser import ParsedLogEntry


@dataclass(frozen=True)
class LogChunk:
    chunk_index: int
    entries: tuple[ParsedLogEntry, ...]
    error_line_numbers: tuple[int, ...]

    @property
    def start_line_number(self) -> int:
        return self.entries[0].line_number

    @property
    def end_line_number(self) -> int:
        return self.entries[-1].line_number

    @property
    def text(self) -> str:
        lines = []
        for entry in self.entries:
            fields = []
            if entry.timestamp is not None:
                fields.append(entry.timestamp)
            if entry.level is not None:
                fields.append(entry.level)
            fields.append(entry.message)
            lines.append(" ".join(fields))
        return "\n".join(lines)


def create_event_chunks(
    entries: list[ParsedLogEntry],
    context_entries: int = 3,
) -> list[LogChunk]:
    """Create merged context chunks around ERROR and CRITICAL log entries."""
    if context_entries < 0:
        raise ValueError("context_entries must be zero or greater")

    error_indices = [
        index
        for index, entry in enumerate(entries)
        if entry.level in {"ERROR", "CRITICAL"}
    ]
    if not error_indices:
        return []

    windows = [
        (
            max(0, error_index - context_entries),
            min(len(entries) - 1, error_index + context_entries),
            entries[error_index].line_number,
        )
        for error_index in error_indices
    ]

    chunks: list[LogChunk] = []
    start_index, end_index, error_line = windows[0]
    error_line_numbers = [error_line]

    for next_start, next_end, next_error_line in windows[1:]:
        if next_start <= end_index + 1:
            end_index = max(end_index, next_end)
            error_line_numbers.append(next_error_line)
            continue

        chunks.append(
            LogChunk(
                chunk_index=len(chunks) + 1,
                entries=tuple(entries[start_index : end_index + 1]),
                error_line_numbers=tuple(error_line_numbers),
            )
        )
        start_index, end_index = next_start, next_end
        error_line_numbers = [next_error_line]

    chunks.append(
        LogChunk(
            chunk_index=len(chunks) + 1,
            entries=tuple(entries[start_index : end_index + 1]),
            error_line_numbers=tuple(error_line_numbers),
        )
    )
    return chunks
