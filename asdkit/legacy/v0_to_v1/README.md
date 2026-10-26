# Migrate to ASDKit 1.0.0

Convert ASDKit 0.0.0 checkpoints for use with ASDKit 1.0.0.
You need the original checkpoint and training configuration.
After installing ASDKit, run the commands below from the repository root.

## Convert a Checkpoint

```bash
uv run python -m asdkit.legacy.v0_to_v1.checkpoint \
  --ckpt-path old-model/checkpoints/last.ckpt \
  --output-path converted-model/checkpoints/last.ckpt
```

The example above reads the training configuration from
`old-model/.hydra/config.yaml`. To use a different configuration path:

```bash
uv run python -m asdkit.legacy.v0_to_v1.checkpoint \
  --ckpt-path old-model.ckpt \
  --config-path training-config.yaml \
  --output-path converted-model.ckpt
```

Use a new output path. The tool leaves the original checkpoint unchanged and does not overwrite existing files.

## Load the Converted Checkpoint

```python
from asdkit.utils.asdkit_utils.restore import restore_plfrontend

frontend = restore_plfrontend(
    ckpt_path="converted-model/checkpoints/last.ckpt", device="cpu"
)
```

The converted model can also be used in extraction recipes.

## Inference Differences

Although the above conversion makes the trained weights usable in v1.0.0, rerunning extraction and scoring introduces these differences:

- **kNN backends:**
  - Source and target references are now searched together. Neighbor selection is unchanged when both old neighbor counts and the new `n_neighbors` are `1`; larger counts can change rankings.
  - Cosine distance replaces Euclidean distance on normalized vectors. For cosine similarity $c$, plain 1-NN scores change from $\sqrt{2-2c}/2$ to $1-c$, preserving rankings and AUC/pAUC. Neighbor averaging and rescaling can change rankings.
- **AE output name:** `AS-recon` is now `AS-main`.

See the [release notes](../../../docs/v1.0.0.md) for other changes.

