from collections import defaultdict
from pathlib import Path

import pandas as pd


def read_incorrect_labels(year: int):
    """
    from https://github.com/nttcslab/dcase2023_task2_baseline_ae/blob/main/datasets/eval_data_list_202*.csv
    """
    path_dict = defaultdict(dict)
    file_path = f"./eval_data_list_{year}.csv"
    with open(file_path, "r") as f:
        for line in f:
            csv_data_list = line.strip().split(",")
            if not csv_data_list[0].endswith(".wav"):
                current_machine = csv_data_list[0]
                continue
            else:
                assert csv_data_list[0].endswith(".wav")
                assert csv_data_list[1].endswith(".wav")
                idx = csv_data_list[1].split("_")[4] == "anomaly"
                path_dict[current_machine][csv_data_list[0]] = idx
    return path_dict


def read_csv_as_dict(file_path):
    data_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            filename, label = line.strip().split(",")
            data_dict[filename] = int(label)
    return data_dict


def read_correct_labels(year: int, evaluator_dir: str):
    """
    2020: https://zenodo.org/records/3951620
    2021: https://github.com/y-kawagu/dcase2021_task2_evaluator
    2022: https://github.com/Kota-Dohi/dcase2022_evaluator
    2023: https://github.com/nttcslab/dcase2023_task2_evaluator
    2024: https://github.com/nttcslab/dcase2024_task2_evaluator
    """
    if year == 2021:
        file_path = f"{evaluator_dir}/dcase2021_task2_evaluator/ground_truth_data"
    elif year == 2022:
        file_path = f"{evaluator_dir}/dcase2022_evaluator/ground_truth_data"
    elif year == 2023:
        file_path = f"{evaluator_dir}/dcase2023_task2_evaluator/ground_truth_data"
    elif year == 2024:
        file_path = f"{evaluator_dir}/dcase2024_task2_evaluator/ground_truth_data"
    label_dict = defaultdict(dict)
    for csv_path in Path(file_path).glob("*.csv"):
        # filename: ground_truth_bearing_section_03_test.csv
        machine = csv_path.name.split("_")[2]
        label_dict[machine] = {**label_dict[machine], **read_csv_as_dict(csv_path)}
    return label_dict


def check_label(year: int, evaluator_dir: str):
    print(f"------------ {year} -----------")
    if year == 2020:
        # 2020: https://zenodo.org/records/3951620
        current_df = pd.read_csv("./eval_data_list_2020.csv")
        correct_df = pd.read_csv(f"{evaluator_dir}/eval_data_list.csv")
        if current_df.equals(correct_df):
            print("Total mismatches for all machines: 0")
    else:
        current_labels = read_incorrect_labels(year)
        correct_labels = read_correct_labels(year, evaluator_dir)
        total_cnt = 0
        for machine in current_labels:
            cnt = 0
            for name, is_anomaly in current_labels[machine].items():
                if correct_labels[machine][name] != is_anomaly:
                    cnt += 1
            print(f"Total mismatches for {machine}: {cnt}")
            total_cnt += cnt
        print(f"Total mismatches for all machines: {total_cnt}")


def main(input_file: str, output_file: str, evaluator_dir: str):
    correct_label_dict = read_correct_labels(2022, evaluator_dir)
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            csv_data_list = line.strip().split(",")
            if not csv_data_list[0].endswith(".wav"):
                current_machine = csv_data_list[0]
                f_out.write(line)
            else:
                assert csv_data_list[0].endswith(".wav")
                assert csv_data_list[1].endswith(".wav")
                is_anomaly = correct_label_dict[current_machine][csv_data_list[0]]
                normal_anomaly_label = "anomaly" if is_anomaly else "normal"
                split_filename_0 = csv_data_list[0].split("_")
                split_filename_1 = csv_data_list[1].split("_")
                # filename_0: section_03_0011.wav
                # filename_1: section_03_source_test_anomaly_0011_vel_21.wav
                modified_full_filename = "_".join(
                    split_filename_1[:4]
                    + [normal_anomaly_label]
                    + [split_filename_0[-1].replace(".wav", "")]
                    + split_filename_1[6:]
                )
                f_out.write(f"{csv_data_list[0]},{modified_full_filename}\n")


if __name__ == "__main__":
    input_file = "eval_data_list_2022.csv"
    output_file = "modified_eval_data_list_2022.csv"
    evaluator_dir = "../../evaluator"
    check_label(2020, evaluator_dir)
    check_label(2021, evaluator_dir)
    check_label(2022, evaluator_dir)
    check_label(2023, evaluator_dir)
    check_label(2024, evaluator_dir)
    main(input_file, output_file, evaluator_dir)


"""
------------ 2020 -----------
Total mismatches for all machines: 0
------------ 2021 -----------
Total mismatches for fan: 0
Total mismatches for gearbox: 0
Total mismatches for pump: 0
Total mismatches for slider: 0
Total mismatches for ToyCar: 0
Total mismatches for ToyTrain: 0
Total mismatches for valve: 0
Total mismatches for all machines: 0
------------ 2022 -----------
Total mismatches for bearing: 0
Total mismatches for fan: 0
Total mismatches for gearbox: 0
Total mismatches for slider: 0
Total mismatches for ToyCar: 296
Total mismatches for ToyTrain: 312
Total mismatches for valve: 0
Total mismatches for all machines: 608
------------ 2023 -----------
Total mismatches for bandsaw: 0
Total mismatches for grinder: 0
Total mismatches for shaker: 0
Total mismatches for ToyDrone: 0
Total mismatches for ToyNscale: 0
Total mismatches for ToyTank: 0
Total mismatches for Vacuum: 0
Total mismatches for all machines: 0
------------ 2024 -----------
Total mismatches for 3DPrinter: 0
Total mismatches for AirCompressor: 0
Total mismatches for Scanner: 0
Total mismatches for ToyCircuit: 0
Total mismatches for HoveringDrone: 0
Total mismatches for HairDryer: 0
Total mismatches for ToothBrush: 0
Total mismatches for RoboticArm: 0
Total mismatches for BrushlessMotor: 0
Total mismatches for all machines: 0
"""
