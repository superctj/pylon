#!/bin/bash

# Stop on errors
set -Eeuo pipefail

dataset_name="tus"
version="small"
instance_name="tabert_cl"
ckpt_name="408_value_cl_epoch_19"
num_samples=10
threshold=0.7

for k in $(seq 10 40 330)
do
    python topk_search_tus.py \
    --dataset_name=${dataset_name} \
    --dataset_dir="/ssd/congtj/data/table-union-search-benchmark/table_csvs/${version}_benchmark/" \
    --query_file="/ssd/congtj/data/table-union-search-benchmark/table_csvs/${version}_groundtruth/recall_groundtruth.csv" \
    --ground_truth_file="/ssd/congtj/data/table-union-search-benchmark/${version}_groundtruth.pkl" \
    --index_dir="/ssd/congtj/tus_artifacts/indexes/${instance_name}/${dataset_name}_${version}/" \
    --output_dir="/home/congtj/pylon_metaspace/pylon/${instance_name}/results/${dataset_name}_${version}/" \
    --ckpt_path="/ssd/congtj/pylon_artifacts/checkpoints/${ckpt_name}.ckpt" \
    --embedding_dim=128 \
    --num_samples=${num_samples} \
    --lsh_threshold=${threshold} \
    --top_k=${k}
done