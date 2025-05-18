import re
from typing import List


def item_match(item: str, patterns: List[str]) -> bool:
    return any(re.fullmatch(p, item) for p in patterns)
