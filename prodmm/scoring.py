"""Score DMS substitutions using wild-type encoder logits."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from Bio import SeqIO

from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer


def read_seq(fasta_file: str | Path) -> str:
    """Read the first FASTA record, matching the original scoring convention."""
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)
    raise ValueError(f"No sequence found in FASTA file: {fasta_file}")


@torch.inference_mode()
def score_mutant(
    mutant: str | Path,
    fasta: str | Path,
    model: torch.nn.Module,
    tokenizer: UniTokenizer,
    tag: str = "cds",
) -> pd.DataFrame:
    """Score substitutions with one model pass and one result transfer.

    Positions are one-based token positions (codons for CDS). The caller
    should put the model in evaluation mode before scoring.
    """
    if tag not in {"cds", "aas", "ncds"}:
        raise ValueError(f"Invalid tag: {tag}")
    sequence = read_seq(fasta)
    df = pd.read_csv(mutant)
    if df.empty:
        df["prediction"] = pd.Series(dtype=float)
        return df

    inputs = tokenizer(
        [f"<{tag}>{sequence}</{tag}>"],
        return_tensors="pt",
        padding=True,
    )
    token_width = 3 if tag == "cds" else 1
    sequence_length = inputs["input_ids"].shape[1] - 2
    vocab = tokenizer.get_vocab()
    rows, positions, wt_ids, mut_ids = [], [], [], []
    for row, mutation in enumerate(df["mutant"]):
        for substitution in mutation.split(":"):
            wild_type, position, mutant_token = (
                substitution[:token_width],
                int(substitution[token_width:-token_width]) - 1,
                substitution[-token_width:],
            )
            if not 0 <= position < sequence_length:
                raise ValueError(f"Mutation position out of range: {substitution}")
            if tag in {"cds", "ncds"}:
                wild_type, mutant_token = wild_type.lower(), mutant_token.lower()
            rows.append(row)
            positions.append(position)
            wt_ids.append(vocab[wild_type])
            mut_ids.append(vocab[mutant_token])

    device = next(model.parameters()).device
    outputs = model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
    )
    logits = outputs.logits[0, 1:-1, :]
    rows = torch.tensor(rows, dtype=torch.long, device=logits.device)
    positions = torch.tensor(positions, dtype=torch.long, device=logits.device)
    wt_ids = torch.tensor(wt_ids, dtype=torch.long, device=logits.device)
    mut_ids = torch.tensor(mut_ids, dtype=torch.long, device=logits.device)
    # log-softmax's normalization cancels in the mutant/wild-type difference.
    differences = logits[positions, mut_ids] - logits[positions, wt_ids]
    predictions = logits.new_zeros(len(df))
    predictions.index_add_(0, rows, differences)
    df["prediction"] = predictions.cpu().tolist()
    return df


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta",
        type=str,
        default="example_data/NCDS_zhang_2009.fasta",
        help="FASTA file containing the wild-type sequence (first record)",
    )
    parser.add_argument(
        "--dms",
        type=str,
        default="example_data/NCDS_zhang_2009.csv",
        help="CSV with a mutant column, e.g. T40A or A39G:C43T",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="example_data/NCDS_zhang_2009_scored.csv",
        help="Destination CSV; existing files are overwritten",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="ncds",
        choices=("cds", "aas", "ncds"),
    )
    parser.add_argument(
        "--device", default="cuda", help="Model device, e.g. cuda or cpu"
    )
    parser.add_argument(
        "--model",
        default="AI4Protein/prodmm_encoder",
        help="Hugging Face model ID or local checkpoint directory",
    )
    args = parser.parse_args()
    from prodmm.encoder.modeling_prolm import ProLMForMaskedLM

    print("Loading model...")
    tokenizer = UniTokenizer()
    model = ProLMForMaskedLM.from_pretrained(args.model)
    model = model.to(args.device).eval()
    df = score_mutant(args.dms, args.fasta, model, tokenizer, args.tag)
    df.to_csv(args.save, index=False)

    if "score" in df.columns:
        from scipy.stats import spearmanr

        print("Spearman correlation", spearmanr(df["score"], df["prediction"]))


if __name__ == "__main__":
    main()
