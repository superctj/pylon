#!/bin/bash

# Stop on errors
set -Eeuo pipefail

dataset_name="pylon"
instance_name="wte_cl"
ckpt_name="420_wte_cl_epoch_9"
num_samples=-1
threshold=0.7

for k in $(seq 10 10 100)
do
    python topk_search_pylon.py \
    --dataset_name=dataset_name \
    --dataset_dir="/ssd/congtj/data/pylon_benchmark/source" \
    --query_file="/ssd/congtj/data/pylon_benchmark/all_queries.txt" \
    --ground_truth_file="/ssd/congtj/data/pylon_benchmark//all_ground_truth.pkl" \
    --index_dir="/ssd/congtj/pylon_artifacts/indexes/${instance_name}/${dataset_name}" \
    --output_dir="/home/congtj/pylon_metaspace/pylon/${instance_name}/results/${dataset_name}" \
    --encoder_path="/ssd/congtj/pylon_artifacts/web_table_embedding_models/web_table_embeddings_combo150.bin" \
    --ckpt_path="/ssd/congtj/pylon_artifacts/web_table_embedding_models/${ckpt_name}.ckpt" \
    --embedding_dim=64 \
    --num_samples=${num_samples} \
    --lsh_threshold=${threshold} \
    --top_k=${k}
done