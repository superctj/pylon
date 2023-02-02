#!/bin/bash

# Stop on errors
set -Eeuo pipefail

dataset_name="santos"
version="labeled"
instance_name="wte_cl"
ckpt_name="420_wte_cl_epoch_9"
num_samples=-1
threshold=0.7

for k in $(seq 10 10 50)
do
    python topk_search_santos_labeled.py \
    --dataset_name=dataset_name \
    --dataset_dir="/ssd/congtj/data/santos_data/benchmark/santos_benchmark/datalake/" \
    --query_dir="/ssd/congtj/data/santos_data/benchmark/santos_benchmark/query/" \
    --ground_truth_file="/home/congtj/pylon_metaspace/santos_baseline/groundtruth/LABELED_benchmark_groundtruth.csv" \
    --index_dir="/ssd/congtj/pylon_artifacts/indexes/${instance_name}/${dataset_name}_${version}/" \
    --output_dir="/home/congtj/pylon_metaspace/pylon/${instance_name}/results/${dataset_name}_${version}/" \
    --encoder_path="/ssd/congtj/pylon_artifacts/web_table_embedding_models/web_table_embeddings_combo150.bin" \
    --ckpt_path="/ssd/congtj/pylon_artifacts/web_table_embedding_models/${ckpt_name}.ckpt" \
    --embedding_dim=64 \
    --num_samples=${num_samples} \
    --lsh_threshold=${threshold} \
    --top_k=${k}
done