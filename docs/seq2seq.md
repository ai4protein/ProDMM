# Seq2seq training and evaluation

Install the optional dependencies from the repository root:

```bash
python -m pip install -e '.[train]'
prodmm-seq2seq --help
```

The command separates training (`--mode train`) from teacher-forced evaluation (`--mode test`). It runs from an installed package and accepts local configurations, pretrained model directories, and Lightning checkpoints. It does not require WandB or TensorBoard logging unless you select those loggers.

## Run a small CPU example

The repository includes a tiny configuration and two synthetic protein/CDS pairs. This checks the software workflow; it is not a biological benchmark or a pretrained model.

```bash
prodmm-seq2seq \
  --mode train \
  --config_path example_data/seq2seq_tiny_config \
  --train_file example_data/seq2seq_tiny.jsonl \
  --valid_file example_data/seq2seq_tiny.jsonl \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --max_steps 2 \
  --accelerator cpu \
  --save_model_dir /tmp/prodmm-demo \
  --save_model_name tiny
```

The output directory contains `best.ckpt` and `last.ckpt` (Lightning checkpoints with optimizer state) and `model/` (the final model configuration and safetensors weights for `from_pretrained()`). Exported model weights are from the end of training; `best.ckpt` is selected by validation loss.

Evaluate the exported model:

```bash
prodmm-seq2seq \
  --mode test \
  --config_path /tmp/prodmm-demo/tiny/model \
  --pretrain_ckpt /tmp/prodmm-demo/tiny/model \
  --test_file example_data/seq2seq_tiny.jsonl \
  --accelerator cpu
```

The report contains loss, target-token accuracy, and perplexity. Padding labels (`-100`) are excluded; epoch metrics are weighted by valid token count. Evaluation runs the supplied data once. These are teacher-forced metrics, where earlier target tokens are supplied to the decoder, rather than metrics for freely generated sequences.

## Prepare your own data

Each JSONL record contains `S` (tagged source), `T` (tagged target), and `L` (sequence length metadata):

```json
{"S":"<aas>MAG<aas2cds></aas>","T":"<cds>ATGGCTGGT</cds>","L":3}
```

The tokenizer preserves task prompt tokens inside a modality segment. CDS targets must have complete codons. The collator adds source boundaries and shifts target labels for the decoder. Keep train, validation, and test records separate in real experiments.

For an existing seq2seq model, supply both `--config_path CHECKPOINT_DIR` and `--pretrain_ckpt CHECKPOINT_DIR`. For initialization from an encoder, supply the desired seq2seq configuration with `--encoder_ckpt ENCODER_DIR`. Configurations must use the prompt vocabulary and matching special-token IDs.

Use `--accelerator gpu --precision bf16-mixed` on compatible hardware. Set `--frozen_encoder true` to freeze encoder parameters. Boolean arguments accept `true` and `false`; passing `false` disables the option.

Resume training with `--trainer_ckpt PATH_TO_LAST_CKPT` and a larger `--max_steps` limit. Add `--max_epochs -1` when you want the step limit to control completion; otherwise, the epoch limit can end the resumed run before any additional optimizer steps. Keep the configuration and data settings consistent with the original run. Resume restores model, optimizer, and scheduler state.

Streaming is supported for a single process with `--streaming true` and a positive `--max_steps`. Multiple data-loader workers can split shards; distributed streaming requires explicit node sharding and is not provided by this command. Non-streaming distributed training uses Lightning's distributed sampler.

## Extend the workflow

- `prodmm/seq2seq/collators/seq2seq.py` handles sequence-to-tensor conversion.
- `prodmm/seq2seq/datamodule.py` owns datasets and loaders.
- `prodmm/seq2seq/lightning_module_finetuning.py` defines `Seq2SeqModule`, optimization, and metrics.
- `finetunes/finetune_seq2seq/finetune_seq2seq.py` parses options and starts the trainer.

The historical `LightningProLM` import is retained as an alias to `Seq2SeqModule`. Its constructor now uses explicit options; it no longer accepts an argparse namespace or owns a test dataset. Callers should provide data through `FinetuningDataModule`.

The model keeps its existing checkpoint names and its default 12-layer decoder. Set `decoder_num_hidden_layers` in a new configuration to use a smaller decoder, as in the supplied CPU example. Existing checkpoints retain their original architecture.

The five task shell scripts are now thin evaluation wrappers. Their arguments are `TEST_JSONL CHECKPOINT_DIR [additional CLI options...]`; they no longer embed private filesystem paths or force CUDA debugging flags. Example:

```bash
bash finetunes/finetune_seq2seq/Reverse-translation.sh \
  my_test.jsonl my_seq2seq_checkpoint --accelerator cpu
```

Older unused launcher flags (such as `--use_decoder`, `--tokenizer_path`, and `--num_training_data`) were removed. Use `prodmm-seq2seq --help` for the supported interface.
