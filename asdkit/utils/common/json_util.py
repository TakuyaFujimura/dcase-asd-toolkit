import json
from pathlib import Path


def write_json(json_path: Path | str, data: dict | list, indent: int = 4) -> None:
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent)


def read_json(json_path: Path | str) -> dict | list:
    with open(json_path) as f:
        data = json.load(f)
    return data
