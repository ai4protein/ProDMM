"""Exercise the public command with a local checkpoint and real file I/O."""

import subprocess
import sys

import pandas as pd
import pytest

from prodmm.encoder.configuration_prolm import ProLMConfig
from prodmm.encoder.modeling_prolm import ProLMForMaskedLM
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer


@pytest.mark.parametrize("entry_point", ["prodmm.scoring", "score_dms"])
def test_scoring_command_with_local_checkpoint(tmp_path, entry_point):
    tokenizer = UniTokenizer()
    config = ProLMConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
    )
    checkpoint = tmp_path / "checkpoint"
    ProLMForMaskedLM(config).save_pretrained(checkpoint)
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(">wild_type\nACGT\n")
    csv = tmp_path / "mutants.csv"
    pd.DataFrame({"mutant": ["A1T", "C2G:T4A"], "score": [0.1, 0.2]}).to_csv(
        csv, index=False
    )
    destination = tmp_path / "scored.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            entry_point,
            "--model",
            str(checkpoint),
            "--fasta",
            str(fasta),
            "--dms",
            str(csv),
            "--save",
            str(destination),
            "--tag",
            "ncds",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    scored = pd.read_csv(destination)
    assert scored.columns.tolist() == ["mutant", "score", "prediction"]
    assert scored["mutant"].tolist() == ["A1T", "C2G:T4A"]
    assert scored["prediction"].notna().all()
    assert "Spearman correlation" in result.stdout
