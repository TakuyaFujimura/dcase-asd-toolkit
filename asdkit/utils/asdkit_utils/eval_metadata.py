import csv
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from urllib.request import urlopen

logger = logging.getLogger(__name__)

BASELINE_REPOSITORY = "nttcslab/dcase2023_task2_baseline_ae"
BASELINE_COMMIT = "f44242ec1f78f6cc34f53f43fb88be1ce5d13d47"
EVALUATOR_2022_REPOSITORY = "Kota-Dohi/dcase2022_evaluator"
EVALUATOR_2022_COMMIT = "de8719b7d35280ff3a39487c9d1003413eef7818"


def get_eval_data_list_path(dcase: str, metadata_dir: Path) -> Path:
    """Resolve the prepared CSV for the configured immutable upstream revision."""
    if dcase not in [f"dcase{year}" for year in range(2020, 2027)]:
        raise ValueError(f"Evaluation metadata is not configured for {dcase}.")
    return metadata_dir / dcase / BASELINE_COMMIT / f"eval_data_list_{dcase[5:]}.csv"


def download_metadata_file(
    repository: str, commit: str, relative_path: str, cache_dir: Path
) -> Path:
    """Cache one upstream file at an immutable Git revision."""
    path = cache_dir / relative_path
    if not path.exists():
        url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative_path}"
        logger.info(f"Downloading metadata: {url}")
        with urlopen(url, timeout=30) as response:
            content = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


def preprocess_dcase2022_eval_rows(rows: List[List[str]], cache_dir: Path) -> None:
    """Correct ToyCar/ToyTrain eval names using official labels and unknown attributes."""
    is_anomaly: Dict[Tuple[str, str], Dict[str, int]] = {}
    is_target: Dict[Tuple[str, str], Dict[str, int]] = {}
    machine = ""
    for row in rows:
        if not row:
            continue
        if not row[0].endswith(".wav"):
            machine = row[0]
            continue
        if machine not in ("ToyCar", "ToyTrain"):
            continue
        source_name = row[0]
        section = source_name.split("_")[1]
        key = (machine, section)
        if key not in is_anomaly:
            for kind, destination in (("data", is_anomaly), ("domain", is_target)):
                label_path = download_metadata_file(
                    repository=EVALUATOR_2022_REPOSITORY,
                    commit=EVALUATOR_2022_COMMIT,
                    relative_path=(
                        f"ground_truth_{kind}/ground_truth_{machine}_section_{section}_test.csv"
                    ),
                    cache_dir=cache_dir / EVALUATOR_2022_COMMIT,
                )
                with label_path.open(encoding="utf-8", newline="") as stream:
                    destination[key] = {
                        name: int(label) for name, label in csv.reader(stream)
                    }
        parts = Path(row[1]).stem.split("_")
        parts[2] = "target" if is_target[key][source_name] else "source"
        parts[4] = "anomaly" if is_anomaly[key][source_name] else "normal"
        parts[5] = Path(source_name).stem.split("_")[-1]
        parts[6:] = ["unknown"]
        row[1] = f"{'_'.join(parts)}.wav"


def preprocess_dcase2024_eval_rows(rows: List[List[str]]) -> None:
    """Insert the missing test field in RoboticArm's destination filenames."""
    machine = ""
    for row in rows:
        if not row:
            continue
        if not row[0].endswith(".wav"):
            machine = row[0]
        elif machine == "RoboticArm":
            parts = row[1].split("_")
            if parts[3] in ["normal", "anomaly"]:
                parts.insert(3, "test")
                row[1] = "_".join(parts)


def prepare_eval_data_list(dcase: str, metadata_dir: Path) -> Path:
    """Download only the requested year's metadata and cache its prepared CSV."""
    csv_path = get_eval_data_list_path(dcase=dcase, metadata_dir=metadata_dir)
    cache_dir = csv_path.parent
    if csv_path.exists():
        return csv_path

    source_path = download_metadata_file(
        repository=BASELINE_REPOSITORY,
        commit=BASELINE_COMMIT,
        relative_path=f"datasets/{csv_path.name}",
        cache_dir=cache_dir,
    )
    with source_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    postprocessors: Dict[str, List[Callable[[List[List[str]]], None]]] = {
        "dcase2022": [partial(preprocess_dcase2022_eval_rows, cache_dir=cache_dir)],
        "dcase2024": [preprocess_dcase2024_eval_rows],
    }
    for postprocess in postprocessors.get(dcase, []):
        postprocess(rows)

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return csv_path
