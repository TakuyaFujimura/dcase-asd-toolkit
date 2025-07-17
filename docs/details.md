## Features
| Name               | Description                                             | Input  | Output |
|--------------------|---------------------------------------------------------| -------|--------|
| asdkit_train       | Train the frontend                                      |  Dataset | Model checkpoint (`model/*/checkpoints/`) 
| asdkit_extract     | Extract features from the frontend                      | Dataset (and checkpoint) | Feature and information (`output/*/<machine>/train_extract.npz` and `test_exctract.npz`) |
| asdkit_score       | Compute anomaly scores from extracted features | Feature and information | Anomaly scores (`output/*/<machine>/train_scores.csv` and `test_scores.csv`) |
| asdkit_evaluate    | Evaluate anomaly scores                                 | Anomaly scores | Evaluation results (`output/*/<machine>/test_evaluate.csv`) |
| asdkit_visualize   | Visualize embeddings                                    | Feature and information | Visualization files (`output/*/<machine>/visualization/`) |
| asdkit_table       | Generate a table of results                             | Evaluation results | Table of results (`output/*/*.csv`) |


## Configuration Items

| Item        | Type             | Description |
|-------------|------------------|-------------|
| `seed`      | `int`            | Random seed. |
| `dcase`     | `str`            | DCASE challenge year. One of: `"dcase2020"`, ..., `"dcase2024"`. |
| `name`      | `str`            | Arbitrary string to identify the name. |
| `version`   | `str`            | Arbitrary string to identify the version. |
| `infer_ver` | `str`            | Arbitrary string to identify the inference version (e.g. epoch). |
| `machine`   | `str`            | Machine type. |
| `model_ver` | `str`            | Arbitrary string to identify the frontend model version. <br> For example, `ae` sets `model_ver` to the machine type because it separately constructs a frontend for each machine type. <br> In contrast, `dis_*` sets it to "all" because it constructs a shared frontend. |
| `ckpt_ver`  | `str`            | Checkpoint identifier. <br> - `epoch_x`: this searches for checkpoints matching `interval_epoch=(x-1)-*.ckpt`. <br> - `last`: this searches for checkpoints matching `last.ckpt`. <br> - `min`/`max` refer to checkpoints with minimum/maximum validation loss.|
| `result_dir`| `Path`           | Path to result directory (relative path from `dcase-asd-toolkit/` or absolute path). |
| `data_dir`  | `Path`           | Path to data directory (relative path from `dcase-asd-toolkit/` or absolute path). |
| `overwrite` | `bool`           | If `True`, overwrite existing files; otherwise, skip the process. |
| `sec`       | `float \| str`   | Duration of audio in seconds. The audio will be padded or truncated to this length in the data loader. <br> If set to a string of the form `dcase_<year>`, the following durations are applied. <br> These values represent the maximum audio length for each DCASE dataset: <br> "dcase2020": 11.0, "dcase2021": 10.0, "dcase2022": 10.0 , "dcase2023": 18.0, "dcase2024": 12.0, "dcase2025": 12.0 |




## Others
- You can modify recipe scripts to customize the process. For example, you can comment out certain commands to skip specific processes.
```bash
asdkit_train experiments="${experiments_train}"
for machine in $machines; do
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
    # asdkit_visualize # Skip visualization
done
asdkit_table
```
