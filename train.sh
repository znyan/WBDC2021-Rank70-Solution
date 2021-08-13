#!/bin/bash
source activate /home/tione/notebook/envs/wbdc

python src/prepare/extract_feed_embeddings.py
python src/prepare/extract_feed_info_features.py
python src/prepare/extract_history_feed_embeddings.py
python src/train/run_mmoe.py