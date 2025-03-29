dcase2sec_dict = {
    "dcase2021": 10.0,
    "dcase2022": 10.0,
    "dcase2023": 18.0,
    "dcase2024": 12.0,
}


def dcase2sec(dcase: str) -> float:
    if dcase in dcase2sec_dict:
        return dcase2sec_dict[dcase]
    else:
        raise ValueError(f"Unexpected sec: {dcase}")
