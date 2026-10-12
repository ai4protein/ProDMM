# Reading and extending the source

## Encoder data flow

A tagged sequence enters `UniTokenizer` in `prodmm/encoder_tokenizer/unitokenizer.py`. The tokenizer separates modalities, splits protein and non-coding DNA into single characters, and splits coding DNA into codons. Vocabulary lookup converts those tokens into IDs. Batch encoding adds boundary tokens and padding and returns an attention mask.

`ProLMConfig` defines model dimensions and attention options. `ProLMForMaskedLM` in `prodmm/encoder/modeling_prolm.py` combines `ProLMModel` with a vocabulary projection. Its outputs include vocabulary logits and the final token embeddings. Supplying labels also computes cross-entropy loss.

`prodmm/scoring.py` reads the wild-type sequence and mutation table, converts substitutions into token indices, runs the encoder once, and gathers the requested logits. Each substitution contributes:

```text
log P(mutant token at position) - log P(wild-type token at position)
= mutant logit - wild-type logit
```

The log-softmax normalization cancels because both tokens refer to the same position. Multiple substitutions in a row are summed. Scores are transferred to the CPU together, then attached to the original table. Scoring disables gradient tracking; its CLI also sets evaluation mode.

## Extension points

| Change | Start here | Verify |
| --- | --- | --- |
| Add a scoring method | `prodmm/scoring.py` | Known expected scores and preserved row order |
| Change sequence parsing | The relevant tokenizer | Case, boundaries, special tokens, padding, and token IDs |
| Add a downstream task | `finetunes/ft_model.py` and `finetunes/finetune.py` | Label shapes, task loss, and a small training batch |
| Change encoder behavior | `prodmm/encoder/modeling_prolm.py` | Tiny-model forward pass and checkpoint compatibility |
| Work on generation | `prodmm/seq2seq/modeling_prolm_seq2seq.py` | Teacher-forced forward pass and generation with the intended checkpoint |
| Extend seq2seq training | `prodmm/seq2seq/lightning_module_finetuning.py` | Train, export/reload, evaluate, and resume on tiny paired data |

Tokenizers load vocabulary files relative to their own modules. Packaging includes these files so installed usage does not depend on the working directory. Preserve vocabulary order: IDs are tied to checkpoint embeddings.

The model class names use the earlier `ProLM` naming. They remain unchanged so imports and checkpoint loading stay compatible. The root `score_dms.py` is a compatibility entry point; put scoring changes in `prodmm/scoring.py`.

## Dependency and validation boundaries

Transformers is pinned to 4.44.2. The models call internal attention and generation helpers, so upgrading that dependency needs import, forward, generation, and checkpoint tests. Training dependencies are optional to keep encoder installation focused.

The local suite verifies CPU behavior with synthetic data, tiny encoder and seq2seq forward passes, and scoring through a local checkpoint. Install `.[train,dev]` to include seq2seq training, export/reload, evaluation, resume, and data-loader checks. The [seq2seq guide](seq2seq.md) explains the public workflow and extension points. These checks do not certify pretrained generation or multi-GPU behavior.

Before claiming pretrained or CUDA compatibility, test the actual checkpoint and hardware. Before claiming a speed improvement, benchmark inference and scoring separately so model execution does not hide the cost being optimized.
