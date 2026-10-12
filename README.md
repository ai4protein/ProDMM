# ProDMM

ProDMM models protein and DNA sequences in a shared vocabulary. This repository contains the source code for *Unveiling Protein-DNA Interdependency: Harnessing Unified Multimodal Sequence Modeling, Understanding and Generation*.

The encoder supports sequence embeddings and mutation scoring. The seq2seq model supports protein–DNA generation research.

## Install

Use Python 3.10 or newer. From a cloned checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The package includes both tokenizer vocabularies and the `prodmm-score-dms` command. Transformers is pinned to 4.44.2 because the custom models use its generation internals. PyTorch 2.5 or newer is required; install the appropriate PyTorch build for your CUDA environment when using a GPU.

Pretrained encoder weights are loaded from [AI4Protein/prodmm_encoder](https://huggingface.co/AI4Protein/prodmm_encoder). The first load downloads weights. You can also pass a local checkpoint directory to `from_pretrained()` or `--model`.

## Generate embeddings

```python
import torch
from prodmm.encoder.modeling_prolm import ProLMForMaskedLM
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer

# Use "cpu" when CUDA is unavailable.
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = UniTokenizer()
model = ProLMForMaskedLM.from_pretrained("AI4Protein/prodmm_encoder").to(device).eval()

sequences = [
    "<aas>MQKHVDTAV</aas>",
    "<cds>ATGCAAAAGCACGTGGACACCGCCGTT</cds>",
    "<ncds>TGCAATGCGCCGTT</ncds><aas>MQKHVDTAV</aas>",
]
inputs = tokenizer(sequences)
with torch.inference_mode():
    outputs = model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
    )
embeddings = outputs.hidden_states  # [batch, padded token count, hidden size]
```

In this model, `hidden_states` contains the final token embeddings. Use `attention_mask` to exclude padding when pooling them. Boundary and separator tokens also have embeddings.

A runnable version is in [examples/embed_sequences.py](examples/embed_sequences.py):

```bash
python examples/embed_sequences.py --device cpu
```

## Sequence formats

| Tag | Sequence | Token unit |
| --- | --- | --- |
| `<aas>…</aas>` | Protein, uppercase letters | One amino acid |
| `<cds>…</cds>` | Coding DNA, either case | One codon (three bases) |
| `<ncds>…</ncds>` | Non-coding DNA, either case | One base |

CDS segments must contain complete codons. Concatenate tagged segments for multimodal input; the tokenizer inserts a separator between them. The encoder adds `<cls>` and `<sep>` around each input. Unsupported sequence characters map to `<unk>`.

## Score mutations

Run from the repository root to use the supplied example data:

```bash
prodmm-score-dms \
  --fasta example_data/NCDS_zhang_2009.fasta \
  --dms example_data/NCDS_zhang_2009.csv \
  --tag ncds \
  --device cpu \
  --save /tmp/ncds_scored.csv
```

Use `--device cuda` for GPU inference. The original `python score_dms.py ...` command also works. For all options, run `prodmm-score-dms --help`.

The FASTA supplies the wild-type sequence from its first record. The CSV must have a `mutant` column:

| Tag | Single substitution | Multiple substitutions |
| --- | --- | --- |
| `aas` | `M1L` | `M1L:G3A` |
| `cds` | `ATG1TAA` | `ATG1TAA:GGT3AAA` |
| `ncds` | `T40A` | `A39G:C43T` |

Positions start at **1**. CDS positions count codons, not bases. Each mutation must refer to a valid position and vocabulary token in the supplied wild-type sequence. The scorer handles substitutions; insertions and deletions are not supported.

The output preserves the input columns and row order and adds `prediction`. A score is the sum of mutant-minus-wild-type log probabilities at the specified positions, using one encoder pass on the unmutated sequence. Larger scores favor the mutant token relative to the wild-type token. If the CSV contains `score`, the command also reports Spearman correlation.

For Python integration, import `score_mutant` from `prodmm.scoring` and pass an encoder in evaluation mode plus a `UniTokenizer`. The model's device determines where inference runs.

For negative-perplexity sequence ranking and the metabolic-pathway checkpoint, see [docs/sequence_scoring.md](docs/sequence_scoring.md).

## Source map

| Path | Purpose |
| --- | --- |
| `prodmm/encoder/` | Encoder configuration and masked-language model |
| `prodmm/encoder_tokenizer/` | Tagged protein/DNA tokenizer and vocabulary |
| `prodmm/scoring.py` | Mutation scoring API and command |
| `prodmm/seq2seq/` | Encoder-decoder model, data loading, and research training code |
| `prodmm/seq2seq_tokenizer/` | Tokenizer with task prompt tokens |
| `finetunes/` | Downstream training scripts and classification head |
| `example_data/` | Small FASTA and DMS inputs |
| `tests/` | Local regression tests without pretrained downloads |

See [docs/development.md](docs/development.md) for data flow and extension points, and [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Fine-tuning and seq2seq

Install the additional research dependencies:

```bash
python -m pip install -e '.[train]'
```

The seq2seq workflow provides explicit training and evaluation modes, model export, and checkpoint resume. Run `prodmm-seq2seq --help` or follow the [CPU example](docs/seq2seq.md) using the supplied tiny configuration and synthetic paired data. See [finetunes/README.md](finetunes/README.md) for downstream-task training and task-specific checkpoint requirements.

## Test and contribute

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m build
```

Tests use fixed logits and tiny randomly initialized models. They verify score equivalence, DNA case handling, mutation positions, tokenizer behavior, encoder and seq2seq forward passes, and the scoring command with a local checkpoint. With the training extra installed, they also run seq2seq training, export/reload, evaluation, and resume. They do not measure pretrained scientific accuracy or GPU performance.

## License

The source is distributed under the [GNU General Public License v3](LICENSE). Model weights and datasets have their own applicable terms.
