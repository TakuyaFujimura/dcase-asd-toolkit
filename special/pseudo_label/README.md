# Pseudo Label

This directory contains experimental scripts for generating pseudo-label JSON files from DCASE training embeddings.
The input files are located at `<results_root_dir> / <dcase> / <teacher_dir> / <machine>/train_extract.npz`.

You can run the entire pseudo-label distillation process using:

```bash
bash jobs/asd/pseudo_label/example.sh
```

Related papers:

- [Pseudo-label distillation for discriminative anomalous sound detection](https://arxiv.org/abs/2607.16678)
- [Improvements of Discriminative Feature Space Training for Anomalous Sound Detection in Unlabeled Conditions](https://ieeexplore.ieee.org/document/10890020)

## Main Programs

#### `main.py`
This conceals the attribute labels for all machines and generates pseudo-labels instead.
The `--config-name` option specifies the configuration file to use. Configuration files are located in `configs/`.


* `machinewise=true`: Performs clustering separately for each machine.
* `machinewise=false`: Performs clustering for all machines together (i.e., without using machine-type labels).

Example:

```bash
python main.py \
  --config-name=kmeans_ratio_08 \
  name=recipe \
  dcase=dcase2023 \
  teacher_dir=raw_beats/0/output/last \
  output_label_name=raw_beats_kmeans_ratio_08
```

#### `main_partial_attribute.py`

This is used for the DCASE 2024 and 2025 datasets, in which some machines have attribute information while others do not.
It uses existing labels for machines with attribute information and generates pseudo-labels only for machines without attribute information.


## Models

* `KMeansModel`: Performs KMeans clustering with a fixed number of clusters.
* `KMeansRatioModel`: Performs KMeans clustering with the number of clusters determined from the ratio to the number of samples.
* `PCA_GEVD_NRFT`: A model for noise-robust feature transformation (NRFT). It requires supplemental embeddings provided in DCASE 2025 (WIP).
