def get_dcase_info(path: str, label: str) -> str | int:
    """Read formatted-path metadata; split denotes train/test/supplemental."""
    # dcase2020/raw/fan/train/section_00_target_train_normal_0009_<attribute>.wav
    filename_parts = path.split("/")[-1].split("_")
    if label == "split":
        return path.split("/")[-2]
    elif label == "machine":
        return path.split("/")[-3]
    elif label == "section":
        return int(filename_parts[1])
    elif label == "is_target":
        assert filename_parts[2] in ["source", "target"]
        return int(filename_parts[2] == "target")
    elif label == "is_normal":
        assert filename_parts[4] in ["normal", "anomaly"]
        return int(filename_parts[4] == "normal")
    elif label == "attr":
        return "_".join(filename_parts[6:]).replace(".wav", "")
    else:
        raise ValueError(f"Unknown label: {label}")
