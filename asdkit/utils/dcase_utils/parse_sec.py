import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


dcase2sec_dict = {
    "dcase2020": 11.0,
    "dcase2021": 10.0,
    "dcase2022": 10.0,
    "dcase2023": 18.0,
    "dcase2024": 12.0,
    "dcase2025": 12.0,
    "dcase2020_fan": 10.0,
    "dcase2020_pump": 10.0,
    "dcase2020_slider": 10.0,
    "dcase2020_ToyCar": 11.0,
    "dcase2020_ToyConveyor": 11.0,
    "dcase2020_valve": 10.0,
    "dcase2021_fan": 10.0,
    "dcase2021_gearbox": 10.0,
    "dcase2021_pump": 10.0,
    "dcase2021_slider": 10.0,
    "dcase2021_ToyCar": 10.0,
    "dcase2021_ToyTrain": 10.0,
    "dcase2021_valve": 10.0,
    "dcase2022_bearing": 10.0,
    "dcase2022_fan": 10.0,
    "dcase2022_gearbox": 10.0,
    "dcase2022_slider": 10.0,
    "dcase2022_ToyCar": 10.0,
    "dcase2022_ToyTrain": 10.0,
    "dcase2022_valve": 10.0,
    "dcase2023_bearing": 10.0,
    "dcase2023_fan": 10.0,
    "dcase2023_gearbox": 10.0,
    "dcase2023_slider": 10.0,
    "dcase2023_ToyCar": 12.0,
    "dcase2023_ToyTrain": 12.0,
    "dcase2023_valve": 10.0,
    "dcase2023_bandsaw": 10.0,
    "dcase2023_grinder": 10.0,
    "dcase2023_shaker": 10.0,
    "dcase2023_ToyDrone": 18.0,
    "dcase2023_ToyNscale": 6.0,
    "dcase2023_ToyTank": 8.0,
    "dcase2023_Vacuum": 15.0,
    "dcase2024_bearing": 10.0,
    "dcase2024_fan": 10.0,
    "dcase2024_gearbox": 10.0,
    "dcase2024_slider": 10.0,
    "dcase2024_ToyCar": 12.0,
    "dcase2024_ToyTrain": 12.0,
    "dcase2024_valve": 10.0,
    "dcase2024_3DPrinter": 10.0,
    "dcase2024_AirCompressor": 10.0,
    "dcase2024_BrushlessMotor": 6.5,
    "dcase2024_HairDryer": 7.0,
    "dcase2024_HoveringDrone": 8.0,
    "dcase2024_RoboticArm": 7.90225,
    "dcase2024_Scanner": 10.0,
    "dcase2024_ToothBrush": 6.0,
    "dcase2024_ToyCircuit": 8.0,
    "dcase2025_bearing": 10.0,
    "dcase2025_fan": 10.0,
    "dcase2025_gearbox": 10.0,
    "dcase2025_slider": 10.0,
    "dcase2025_ToyCar": 12.0,
    "dcase2025_ToyTrain": 12.0,
    "dcase2025_valve": 10.0,
    "dcase2025_AutoTrash": 6.0,
    "dcase2025_BandSealer": 10.0,
    "dcase2025_CoffeeGrinder": 5.0,
    "dcase2025_HomeCamera": 6.0,
    "dcase2025_Polisher": 10.0,
    "dcase2025_ScrewFeeder": 10.0,
    "dcase2025_ToyPet": 10.0,
    "dcase2025_ToyRCCar": 7.0,
}


def dcase2sec(dcase: str) -> float:
    if dcase in dcase2sec_dict:
        return dcase2sec_dict[dcase]
    else:
        raise ValueError(f"Unexpected sec: {dcase}")


def parse_sec_inplace(cfg: Any):
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == "sec" and isinstance(v, str) and v.startswith("dcase202"):
                cfg[k] = dcase2sec(v)
                logger.info(f"sec: {v} -> {cfg[k]}")
            elif isinstance(v, (dict, list)):
                parse_sec_inplace(v)
    elif isinstance(cfg, list):
        for v in cfg:
            parse_sec_inplace(v)


def parse_sec_cfg(cfg: dict) -> dict:
    cfg_new = copy.deepcopy(cfg)
    parse_sec_inplace(cfg_new)
    return cfg_new
