#!/bin/bash
# Resumes prepare_audio.sh from the PCA step onward.
# Run this when feature extraction and clustering already completed
# but PCA/apply_pca/merge_clusters/mean_pool did not finish.

set -e

FAIRSEQ_ROOT="/home/ajegetina/wav2vec_unsupervised/fairseq_"
CLUSTERING_DIR="/home/ajegetina/wav2vec_unsupervised/data/clustering/librispeech"
MODEL="/home/ajegetina/wav2vec_unsupervised/pre-trained/wav2vec_vox_new.pt"
VENV="/home/ajegetina/wav2vec_unsupervised/venv"
CHECKPOINT_FILE="/home/ajegetina/wav2vec_unsupervised/data/checkpoints/librispeech/progress.checkpoint"

source "$VENV/bin/activate"
export FAIRSEQ_ROOT

DIM=512
SPLITS=(train valid test)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running PCA..."
python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/pca.py" \
  "$CLUSTERING_DIR/train.npy" --output "$CLUSTERING_DIR/pca" --dim $DIM

for split in "${SPLITS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] apply_pca for $split..."
  python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/apply_pca.py" \
    "$CLUSTERING_DIR" --split "$split" \
    --save-dir "$CLUSTERING_DIR/precompute_pca${DIM}" \
    --pca-path "$CLUSTERING_DIR/pca/${DIM}_pca" --batch-size 32

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] merge_clusters for $split..."
  python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py" \
    "$CLUSTERING_DIR/precompute_pca${DIM}" \
    --cluster-dir "$CLUSTERING_DIR/CLUS128" \
    --split "$split" \
    --save-dir "$CLUSTERING_DIR/precompute_pca${DIM}_cls128_mean" \
    --pooling mean

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] mean_pool for $split..."
  python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/mean_pool.py" \
    "$CLUSTERING_DIR/precompute_pca${DIM}_cls128_mean" \
    --save-dir "$CLUSTERING_DIR/precompute_pca${DIM}_cls128_mean_pooled" \
    --split "$split"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Marking prepare_audio as completed..."
echo "prepare_audio:COMPLETED" >> "$CHECKPOINT_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] resume_prepare_audio DONE"
