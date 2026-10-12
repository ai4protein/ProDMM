#!/usr/bin/env bash
# Evaluate a supplied checkpoint on paired source/target JSONL records.
set -euo pipefail
if [ "$#" -lt 2 ]; then
  printf 'Usage: %s TEST_JSONL CHECKPOINT_DIR [additional CLI options...]\n' "$0" >&2
  exit 2
fi
test_file=$1
checkpoint=$2
shift 2
python3 -m finetunes.finetune_seq2seq.finetune_seq2seq \
  --mode test \
  --test_file "$test_file" \
  --config_path "$checkpoint" \
  --pretrain_ckpt "$checkpoint" \
  --task_type prot_cds \
  --cds_gen true \
  "$@"
