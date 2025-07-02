import logging
from pathlib import Path


def set_logging(dst_dir: Path, filename) -> None:
    logging.basicConfig(
        filename=dst_dir / f"{Path(filename).stem}.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
