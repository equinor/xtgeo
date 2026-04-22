from typing import Generator, TextIO

Line = list[str]


class TextParser:
    """
    Static light-weight methods for text parsing,
    to support reading of text files with headers, numerical data and comments.
    """

    @staticmethod
    def iter_nonempty_lines(
        stream: TextIO,
    ) -> Generator[Line, None, None]:
        """
        Lazily iterate over lines from a TextIO, yielding lists of strings
        typically representing words and numbers.
        Filters out empty lines.
        """
        for line in stream:
            split_line = line.strip().split()

            if not split_line:
                continue

            yield split_line

    @staticmethod
    def is_comment(line: Line, comment_prefixes: list[str]) -> bool:
        """
        Check if a line is a comment line,
        i.e. starting with any of the comment prefixes.
        """
        return any(line[0].startswith(prefix) for prefix in comment_prefixes)

    @staticmethod
    def starts_with_prefix(line: Line, prefix: str) -> bool:
        """
        Check if line starts with a specific prefix, typically being a keyword
        indicating a specific section of the file and preceeded with
        a set of numeric values.
        """
        return line[0].startswith(prefix)
