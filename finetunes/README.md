# Fine-tuning ProDMM

Install optional training dependencies from the repository root:

```bash
python -m pip install -e '.[train]'
```

## Seq2seq

Use `prodmm-seq2seq --mode train` to train or `--mode test` for teacher-forced evaluation. The [seq2seq guide](../docs/seq2seq.md) includes a runnable CPU example, input records, export/reload instructions, and checkpoint resume.

The source separates the collator, data module, Lightning module, and CLI. Tests cover training, export, reload, evaluation, resume, streaming, and padding-aware metrics with tiny local data. Pretrained biological accuracy, free generation, and multi-GPU execution still require experiment-specific validation.

Task shell scripts under `finetune_seq2seq/` accept `TEST_JSONL CHECKPOINT_DIR [additional CLI options...]` and invoke the same evaluation command.

## Encoder downstream tasks

`finetune.py` contains the task collator, Lightning module, dataset loader, and argument parser. `ft_model.py` defines the sequence-classification head. Inspect arguments with:

```bash
python -m finetunes.finetune --help
```

Training expects JSONL records with `S` (sequence), `L` (length), `label`, and `value`; the task determines which target is used. Regression uses `value`, classification uses `label`, and multilabel classification uses a list of labels.

`finetune.sh` remains a research example for localization, regression, and multilabel tasks. Its checkpoint and `data/CaLM/...` input paths are environment-specific; adapt them to your dataset before running. The encoder collator is regression-tested, but the full downstream-task training launcher is not included in the end-to-end smoke test.
