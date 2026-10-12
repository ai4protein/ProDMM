"""CLI parsing remains usable without the optional training dependencies."""

import pytest

from finetunes.finetune_seq2seq.finetune_seq2seq import parse_args


def test_explicit_false_is_false():
    args = parse_args(
        [
            "--config_path",
            "checkpoint",
            "--test_file",
            "test.jsonl",
            "--frozen_encoder",
            "false",
            "--streaming",
            "false",
            "--cds_gen",
            "true",
        ]
    )
    assert args.frozen_encoder is False
    assert args.streaming is False
    assert args.cds_gen is True


def test_train_requires_both_data_splits():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "train", "--config_path", "checkpoint"])
