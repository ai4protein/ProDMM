# Perplexity-based sequence scoring

The original encoder workflow ranks unmasked sequences by negative perplexity: `-exp(loss)`, where `loss` is the model's mean token cross-entropy. Larger values are favored. This differs from the substitution log-odds score produced by `prodmm-score-dms`.

```python
import torch
from prodmm.encoder.modeling_prolm import ProLMForMaskedLM
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer


@torch.inference_mode()
def score_sequence(model, tokenizer, sequence):
    device = next(model.parameters()).device
    inputs = tokenizer([sequence])
    input_ids = inputs["input_ids"].to(device)
    output = model(
        input_ids=input_ids,
        attention_mask=inputs["attention_mask"].to(device),
        labels=input_ids,
    )
    return -output.loss.exp().item()


tokenizer = UniTokenizer()
model = ProLMForMaskedLM.from_pretrained("AI4Protein/prodmm_encoder").eval()
# Call model.to("cuda") to use a GPU.
sequences = [
    "<cds>ATGGCTGGT</cds>",
    "<cds>ATGGCCGGC</cds>",
]
scores = [score_sequence(model, tokenizer, sequence) for sequence in sequences]
print(scores)
```

This preserves the original single-sequence convention, including boundary tokens in the loss. Do not use its single returned loss as per-sequence scores for a padded batch: that loss is averaged over the batch and includes labels for padding unless explicitly masked.

For the original metabolic-pathway workflow, load `AI4Protein/prodmm_encoder_antismash_tuned` and concatenate tagged non-coding DNA, coding DNA, or protein segments in pathway order. Score each complete pathway separately using the same function.
