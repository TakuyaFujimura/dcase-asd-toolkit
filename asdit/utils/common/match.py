import re
from typing import List


def re_match_any(patterns: List[str], string: str) -> bool:
    """
    Check if any pattern matches the string.
    """
    return any(re.fullmatch(p, string) for p in patterns)
