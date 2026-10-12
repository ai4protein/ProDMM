"""Check scoring against the original log-probability definition."""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer
from prodmm.scoring import read_seq, score_mutant


class FixedLogitsModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        assert not torch.is_grad_enabled()
        assert input_ids.device == self.logits.device
        assert attention_mask.shape == input_ids.shape
        return SimpleNamespace(logits=self.logits.unsqueeze(0))


@pytest.mark.parametrize(
    "tag,sequence,mutants",
    [
        ("aas", "MAG", ["M1L", "M1L:G3*", "A2A", "M1L"]),
        ("cds", "ATGGCTGGT", ["ATG1TAA", "ATG1TAA:GGT3AAA", "GCT2GCT"]),
        ("ncds", "ACGT", ["A1T", "A1T:T4A", "C2C", "a1t"]),
    ],
)
def test_scores_match_log_probability_odds(tmp_path, tag, sequence, mutants):
    tokenizer = UniTokenizer()
    fasta = tmp_path / "wild_type.fasta"
    fasta.write_text(f">wild_type\n{sequence}\n")
    csv = tmp_path / "mutants.csv"
    pd.DataFrame({"mutant": mutants, "score": range(len(mutants))}).to_csv(
        csv, index=False
    )
    token_count = len(sequence) // (3 if tag == "cds" else 1)
    logits = torch.randn(
        token_count + 2, len(tokenizer), generator=torch.Generator().manual_seed(7)
    )
    model = FixedLogitsModel(logits)

    result = score_mutant(csv, fasta, model, tokenizer, tag)

    log_probs = logits[1:-1].log_softmax(-1)
    expected = []
    width = 3 if tag == "cds" else 1
    for mutant in mutants:
        score = 0
        for substitution in mutant.split(":"):
            wt, mut = substitution[:width], substitution[-width:]
            position = int(substitution[width:-width]) - 1
            if tag != "aas":
                wt, mut = wt.lower(), mut.lower()
            score += (
                log_probs[position, tokenizer.stoi(mut)]
                - log_probs[position, tokenizer.stoi(wt)]
            )
        expected.append(score.item())

    assert result["prediction"].tolist() == pytest.approx(expected, abs=1e-6)
    assert result["mutant"].tolist() == mutants
    assert result["score"].tolist() == list(range(len(mutants)))
    assert model.calls == 1
    assert model.logits.grad is None


@pytest.mark.parametrize("mutation", ["M0A", "M-1A", "M4A"])
def test_out_of_range_positions_fail_before_inference(tmp_path, mutation):
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">wild_type\nMAG\n")
    csv = tmp_path / "mutants.csv"
    pd.DataFrame({"mutant": [mutation]}).to_csv(csv, index=False)
    tokenizer = UniTokenizer()
    model = FixedLogitsModel(torch.zeros(5, len(tokenizer)))
    with pytest.raises(ValueError, match="out of range"):
        score_mutant(csv, fasta, model, tokenizer, "aas")
    assert model.calls == 0


def test_empty_csv_does_not_run_model(tmp_path):
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">wild_type\nMAG\n")
    csv = tmp_path / "mutants.csv"
    pd.DataFrame(columns=["mutant", "score"]).to_csv(csv, index=False)
    tokenizer = UniTokenizer()
    model = FixedLogitsModel(torch.zeros(5, len(tokenizer)))
    result = score_mutant(csv, fasta, model, tokenizer, "aas")
    assert result.empty
    assert "prediction" in result
    assert model.calls == 0


def test_empty_fasta_has_clear_error(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    with pytest.raises(ValueError, match="No sequence"):
        read_seq(fasta)


def test_invalid_tag_has_clear_error():
    with pytest.raises(ValueError, match="Invalid tag"):
        score_mutant("unused.csv", "unused.fasta", None, None, "dna")
