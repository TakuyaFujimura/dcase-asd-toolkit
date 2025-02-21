### How to use
1. Specify `<data_dir>` and `<dcase>` in `run.sh`.
- `<data_dir>`: A relative path from `run.sh` or an absolute path
- `<dcase>`: One of the "dcase2021", "dcase2022", "dcase2023", "dcase2024".

    (Currently, "dcase2020" is not supported.)

2. Execute `run.sh`.


### Result
```bash
<data_dir>
├── original
│   ├── <dcase>
│   └── ...
└── formatted
    ├── <dcase>
    └── ...
```
