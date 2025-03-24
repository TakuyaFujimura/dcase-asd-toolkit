## Common Configuration Items
- seed (int): Random seed
- dcase (str): ["dcase2020", ..., "dcase2024"]
- name (str): Arbitrary string used to identify the name
- version (str): Arbitrary string used to identify the version
- infer_ver (str): Arbitrary string used to identify the inference version
- machine (str): Machine type
- model_ver (str): Arbitrary string used to identify the model version. 
In the ae_baseline, model_ver is set to machine type because ae_baseline constructs machine-wise ASD system. In the dis_baseline, it is set to "all" because dis_baseline constructs shared single ASD system.
- ckpt_ver (str): String used to identify the checkpoint version. `epoch_??`, `last`, `min`, and `max` are acceptable, where `min` and `max` are the checkpoint with the minimum and maximum validation loss, respectively.
- result_dir (Path): result directory path from `asdit`
- data_dir (Path): data directory path from `asdit`
- overwrite (bool): If True, the existing files will be overwritten. Otherwise, the process will be skipped.
<!-- - model (asdit.utils.config_class.ModelConfig): Configuration of frontend -->
<!-- - trainer (Dict[str, Any]): Configuration of pl.Trainer -->
<!-- - label_dict_path (Dict[str, Path]): Label file path -->
<!-- - datamodule (asdit.utils.config_class.DMSplitConfig): Configuration of DataModule -->

## frontend
- hoge


## Extract
This extracts information from audio file by using frontend.
Output file `*_extract.csv` can include path, section, is_normal, is_target, and outputs of frontend.

<details><summary>Configuration Items</summary>

- resume_or_scratch (str): ["resume", "scratch"]
- frontend_cfg (Dict[str, Any]): Required when resume_or_scratch is "scratch".
- ckpt_ver (str): Required when resume_or_scratch is "resume"
- model_ver (str): Required when resume_or_scratch is "resume"
- extract_items (List[str]): A list of `re` patterns used to select extracted items. These patterns will be combined with the default items (path, section, is_normal, and is_target).
</details>

<details><summary>Extract features from pre-trained frontend</summary>

The model will be resumed from the checkpoint file of `ckpt_ver` version in `<result_dir>/<name>/<dcase>/<version>/model/<model_ver>/checkpoints`.

Configuration items:
- resume_or_scratch (str): "resume"
- ckpt_ver (str)
- model_ver (str)

</details>


<details><summary>Extract features from training-free frontend</summary>

Training-free frontend (e.g., simple audio feature extractor) will be constructed based on the configuration of `frontend_cfg`. 

Configuration items:
- resume_or_scratch (str): "scratch"
- frontend_cfg (Dict[str, Any]): Please write the configuration of the model used for feature extraction.

</details>

## Score
This loads `*_extract.csv` and calculates anomaly scores.

<details><summary>Configuration Items</summary>

- backend (List[Dict[str, Any]]): A list of backend configurations.
</details>

## Evaluate
This loads `*_score.csv` and calculates evaluation metrics per machine.
Metric name is in the format of `<section>_<domain>_<auc/pauc>`.
Additionally, official metrics are automatically calculated based on the `dcase`.

<details><summary>Domain</summary>

- s: AUC using the normal and anomalous sounds in the source domain
- t: AUC using the normal and anomalous sounds in the target domain
- smix: AUC using the normal and anomalous sounds in the source domain and the anomalous sounds in the target domain
- tmix: AUC using the normal and anomalous sounds in the target domain and the anomalous sounds in the source domain
- mix: AUC using the normal and anomalous sounds in the source and target domains
</details>

<details><summary>Configuration Items</summary>

- hmean_cfg_dict (Dict[str, List[str]]): Additional configuration for harmonic mean calculation. Please specify the `<domain>_<auc/pauc>` to be used for harmonic mean calculation. Do not include `section` because it is automatically added based on the official dev and eval split.
</details>


