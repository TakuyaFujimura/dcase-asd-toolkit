# Copyright 2024 Takuya Fujimura

import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from asdkit.datasets import PLDataModule
from asdkit.frontends.base import BaseFrontend
from asdkit.utils.asdkit_utils.path import get_extraction_marker
from asdkit.utils.asdkit_utils.restore import get_ckpt_path, restore_plfrontend
from asdkit.utils.common import instantiate, re_match_any
from asdkit.utils.common.json_util import read_json
from asdkit.utils.config_class import MainExtractConfig
from asdkit.utils.config_class.extract_config import ExtractDMConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractData:
    """Extraction results and frontend metadata for one split."""

    extract_dict: Dict[str, Any]
    metadata: Dict[str, Any]


def load_extract_data(output_dir: Path, split: str) -> ExtractData:
    directory = output_dir / f"{split}_extract"
    if not directory.is_dir():
        raise FileNotFoundError(f"Extraction directory not found: {directory}")
    if not get_extraction_marker(directory).is_file():
        raise ValueError(f"Incomplete extraction: {directory}")
    extract_dict = {}
    for path in sorted(directory.rglob("*.npy")):
        if not path.is_file():
            continue
        key = str(path.relative_to(directory).with_suffix(""))
        extract_dict[key] = np.load(path, mmap_mode="r", allow_pickle=False)

    # Frontend metadata
    metadata_path = output_dir / f"{split}_extract.metadata.json"
    metadata = read_json(json_path=metadata_path) if metadata_path.exists() else {}
    assert isinstance(metadata, dict)
    return ExtractData(extract_dict=extract_dict, metadata=metadata)


class NpyExtractionWriter:
    """Stream numeric arrays to disk and mark completion only on success.

    String arrays are collected separately so later, longer paths cannot be
    truncated by the dtype of the first batch. Keys become relative npy paths,
    including subdirectories. A .done marker in the directory is published
    only after all arrays have been saved. Re-extraction resets this directory.
    """

    def __init__(self, directory: Path, num_samples: int) -> None:
        self.directory = directory
        self.num_samples = num_samples
        self.arrays: Dict[str, np.memmap] = {}
        self.strings: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.files: Dict[str, str] = {}
        self.offset = 0
        self.marker = get_extraction_marker(directory)
        self.marker.unlink(missing_ok=True)
        # A completed directory must contain only this run's selected keys.
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    def append(self, outputs: Dict[str, Any]) -> None:
        values = {
            key: (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value)
            )
            for key, value in outputs.items()
        }
        if not values:
            raise ValueError("No extraction outputs selected for npy storage")
        if any(value.ndim == 0 for value in values.values()):
            raise ValueError("Extraction arrays must have a sample dimension")
        count = len(next(iter(values.values())))
        end = self.offset + count
        if end > self.num_samples or any(len(v) != count for v in values.values()):
            raise ValueError("Extraction sample count does not match the loader")
        if self.files and values.keys() != self.files.keys():
            raise ValueError("Selected extraction keys must agree across batches")
        for key, value in values.items():
            if value.dtype.hasobject:
                raise ValueError(f"Object arrays cannot be stored safely: {key}")
            if key not in self.files:
                if (
                    not isinstance(key, str)
                    or not key
                    or Path(key).is_absolute()
                    or any(part in ("", ".", "..") for part in key.split("/"))
                ):
                    raise ValueError(
                        f"Invalid extraction key path: {key!r}. "
                        "Keys are used as file paths and must be relative, "
                        "with no empty, '.' or '..' components."
                    )
                self.files[key] = f"{key}.npy"
                (self.directory / self.files[key]).parent.mkdir(
                    parents=True, exist_ok=True
                )
            if value.dtype.kind in "US":
                self.strings[key].append(value)
                continue
            if key in self.strings:
                raise ValueError(f"Extraction dtype changed across batches: {key}")
            if key not in self.arrays:
                self.arrays[key] = np.lib.format.open_memmap(
                    self.directory / self.files[key],
                    mode="w+",
                    dtype=value.dtype,
                    shape=(self.num_samples, *value.shape[1:]),
                )
            array = self.arrays[key]
            if array.dtype != value.dtype or array.shape[1:] != value.shape[1:]:
                raise ValueError(
                    f"Extraction dtype or shape changed across batches: {key}"
                )
            array[self.offset : end] = value
        self.offset = end

    def finish(self) -> None:
        if self.offset != self.num_samples:
            raise ValueError(
                f"Expected {self.num_samples} samples, extracted {self.offset}"
            )
        for key, batches in self.strings.items():
            if key in self.arrays:
                raise ValueError(f"Extraction dtype changed across batches: {key}")
            np.save(
                self.directory / self.files[key],
                np.concatenate(batches),
                allow_pickle=False,
            )
        self.close()
        self.marker.touch()

    def close(self) -> None:
        for array in self.arrays.values():
            array.flush()
            array._mmap.close()
        self.arrays.clear()


def get_num_samples(dataloader: DataLoader) -> int:
    """Count output rows for a standard loader with one row per sampled item.

    Custom batch samplers and disabled automatic batching are unsupported.
    The collator and frontend must preserve the sampled row count.
    """
    if dataloader.batch_size is None:
        raise ValueError(
            "npy extraction requires a standard sampler and batch_size; "
            "custom batch samplers are not supported"
        )
    if dataloader.drop_last:
        return len(dataloader) * dataloader.batch_size
    return len(dataloader.sampler)


def create_extraction_loaders(
    datamodule: Dict[str, ExtractDMConfig],
) -> Dict[str, DataLoader]:
    dataloader_dict = {}
    for split, dm_cfg in datamodule.items():
        if dm_cfg.dataloader.get("shuffle") is not False:
            logger.warning(f"cfg.datamodule.{split}.dataloader.shuffle is not False.")
        if dm_cfg.collator.get("shuffle") is not False:
            logger.warning(f"cfg.datamodule.{split}.collator.shuffle is not False.")
        dataloader_dict[split] = PLDataModule.get_loader(dm_config=dm_cfg)
    return dataloader_dict


def create_extraction_frontend(cfg: MainExtractConfig) -> BaseFrontend:
    """Restore or instantiate the frontend without constructing data loaders."""
    if cfg.frontend_source_type == "checkpoint":
        return restore_plfrontend(ckpt_path=get_ckpt_path(cfg), device=cfg.device)
    if cfg.frontend_source_type == "config":
        if cfg.frontend_config is None:
            raise ValueError(
                "frontend_config must be specified when frontend_source_type is config"
            )
        return instantiate(cfg.frontend_config)
    raise ValueError(f"Unexpected frontend_source_type: {cfg.frontend_source_type}")


def select_extraction_outputs(
    batch: Dict[str, Any],
    frontend_output_dict: Dict[str, Any],
    extract_items: List[str],
) -> Dict[str, Any]:
    """Merge batch metadata and frontend outputs, rejecting ambiguous keys."""
    overlap = frontend_output_dict.keys() & batch.keys()
    if overlap:
        raise ValueError(
            "Extraction output conflicts with batch keys: " f"{sorted(overlap)}"
        )
    output_dict = {**frontend_output_dict, **batch}
    return {
        key: value
        for key, value in output_dict.items()
        if re_match_any(patterns=extract_items, string=key)
    }


def extract_batches(
    dataloader: Iterable[Dict[str, Any]],
    frontend: BaseFrontend,
    device: str,
    extract_items: List[str],
    writer: NpyExtractionWriter,
) -> None:
    """Extract and stream selected outputs without retaining full numeric arrays."""
    expected_keys: Optional[Set[str]] = None
    if isinstance(frontend, torch.nn.Module):
        frontend.to(device)
        frontend.eval()

    for batch in tqdm(dataloader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.no_grad():
            frontend_output_dict = frontend.extract(batch=batch)
            output_dict = select_extraction_outputs(
                batch=batch,
                frontend_output_dict=frontend_output_dict,
                extract_items=extract_items,
            )
            selected_keys = set(output_dict)
            if expected_keys is None:
                expected_keys = selected_keys
            elif selected_keys != expected_keys:
                raise ValueError(
                    "Selected extraction keys must agree across batches to preserve sample alignment"
                )
            writer.append(output_dict)
