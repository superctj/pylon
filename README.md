# Pylon
This repo includes the codebase and data of paper [Pylon: Semantic Table Union Search in Data Lakes](https://arxiv.org/abs/2301.04901).

## Environment Setup
    conda env create -f pylon_environment.yml

## Models & Data
Model checkpoints are available at our [Google Drive](https://drive.google.com/drive/folders/1ZeRF6KRuqWUw87AO8Um-2VtI43Rpm0yb?usp=share_link).

`pylon_benchmark.tar.gz` contains the benchmark we created from a real-world corpus ([GitTables](https://gittables.github.io)) for semantic table union search.

## Evaluation
To run the evaluation, follow the steps below:
1. ```cd wte_cl```
2. Change paths as appropriate in bash scripts under `scripts/`
3. Run, for example, the evaluation on the Pylon benchmark ```./scripts/run_wte_cl_pylon.sh```

## Pre-training
To pre-train an embedding model, follow the steps below:
1. ```cd wte_cl/wte_cl_training```
2. Change paths and hyperparameters as appropriate in `train_model.py`
3. ```python train_model.py```

## Citation
If you find our work useful or related to yours, please cite our paper with the entry below:

```
@article{DBLP:journals/corr/abs-2301-04901,
  author       = {Tianji Cong and
                  Fatemeh Nargesian and
                  H. V. Jagadish},
  title        = {Pylon: Semantic Table Union Search in Data Lakes},
  journal      = {CoRR},
  volume       = {abs/2301.04901},
  year         = {2023}
}
```

## What's the Tea? :tea:
> TLDR: One paper with the same idea as ours is published in VLDB 2023. The first author, she *was* a friend.
