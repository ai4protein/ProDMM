"""Verify the optional training workflow with tiny local models and data."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lightning")
pytest.importorskip("datasets")

from finetunes.finetune import BertCollatorFinetune
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer
from prodmm.seq2seq.collators import Seq2SeqCollator
from prodmm.seq2seq.configuration_prolm import ProLMConfig
from prodmm.seq2seq.datamodule import FinetuningDataModule
from prodmm.seq2seq.lightning_module_finetuning import Seq2SeqModule
from prodmm.seq2seq_tokenizer.prompttokenizer import PromptTokenizer


@pytest.fixture
def paired_data(tmp_path):
    records = [
        {"S": "<aas>MA<aas2cds></aas>", "T": "<cds>ATGGCT</cds>", "L": 2},
        {"S": "<aas>M<aas2cds></aas>", "T": "<cds>ATG</cds>", "L": 1},
    ]
    path = tmp_path / "paired.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path, records


@pytest.fixture
def tiny_config(tmp_path):
    tokenizer = PromptTokenizer()
    config = ProLMConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        decoder_num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
    path = tmp_path / "config"
    config.save_pretrained(path)
    return path


def test_collator_uses_single_boundaries_and_masks_target_padding(paired_data):
    _, records = paired_data
    tokenizer = PromptTokenizer()
    batch = Seq2SeqCollator(tokenizer)(records)
    assert batch["input_ids"][0].tolist() == tokenizer.encode(records[0]["S"])
    assert batch["labels"][0].tolist() == [
        tokenizer.stoi("atg"),
        tokenizer.stoi("gct"),
        tokenizer.eos_token_id,
    ]
    assert batch["labels"][1].tolist() == [
        tokenizer.stoi("atg"),
        tokenizer.eos_token_id,
        -100,
    ]


def test_encoder_downstream_collator_does_not_duplicate_boundaries():
    tokenizer = UniTokenizer()
    batch = BertCollatorFinetune(tokenizer)([{"S": "ATGGCT", "value": 0.5}])
    assert batch["input_ids"][0].tolist() == tokenizer.encode("<cds>ATGGCT</cds>")


@pytest.mark.parametrize("streaming", [False, True])
def test_data_module_respects_stage_and_streaming(paired_data, streaming):
    path, _ = paired_data
    data = FinetuningDataModule(
        PromptTokenizer(),
        valid_file=str(path),
        test_file=str(path),
        streaming=streaming,
    )
    data.setup("validate")
    assert not hasattr(data, "_trainset")
    assert next(iter(data.val_dataloader()))["labels"].shape == (2, 3)
    data.setup("test")
    assert next(iter(data.test_dataloader()))["input_ids"].shape[0] == 2


def test_metrics_exclude_padding_and_weight_by_tokens(tiny_config):
    module = Seq2SeqModule(tiny_config)

    class FixedModel(torch.nn.Module):
        def forward(self, labels):
            logits = torch.zeros(*labels.shape, 100)
            logits[..., 0] = 1
            return SimpleNamespace(
                loss=labels.new_tensor(2.0, dtype=torch.float), logits=logits
            )

    module.model = FixedModel()
    module.test_step({"labels": torch.tensor([[0, 0, -100]])}, 0)
    module.test_step({"labels": torch.tensor([[1, -100, -100]])}, 1)
    assert module.test_loss.compute().item() == pytest.approx(2)
    assert module.test_accuracy.compute().item() == pytest.approx(2 / 3)


def test_train_export_and_evaluate_cli(tmp_path, tiny_config, paired_data):
    path, _ = paired_data
    command = [sys.executable, "-m", "finetunes.finetune_seq2seq.finetune_seq2seq"]
    trained = subprocess.run(
        command
        + [
            "--mode",
            "train",
            "--config_path",
            str(tiny_config),
            "--train_file",
            str(path),
            "--valid_file",
            str(path),
            "--train_batch_size",
            "1",
            "--eval_batch_size",
            "1",
            "--max_steps",
            "2",
            "--accelerator",
            "cpu",
            "--save_model_dir",
            str(tmp_path),
            "--save_model_name",
            "run",
            "--log_steps",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert trained.returncode == 0, trained.stderr
    export = tmp_path / "run" / "model"
    assert (export / "config.json").is_file()
    assert (export / "model.safetensors").is_file()
    assert (tmp_path / "run" / "last.ckpt").is_file()
    assert "max_steps=2" in trained.stderr
    # Verify export/reload preserves every trained parameter, including tied aliases.
    saved = torch.load(
        tmp_path / "run" / "last.ckpt", map_location="cpu", weights_only=False
    )
    reloaded = Seq2SeqModule(export, checkpoint=export)
    for name, parameter in reloaded.model.state_dict().items():
        torch.testing.assert_close(parameter, saved["state_dict"][f"model.{name}"])
    evaluated = subprocess.run(
        command
        + [
            "--mode",
            "test",
            "--config_path",
            str(export),
            "--pretrain_ckpt",
            str(export),
            "--test_file",
            str(path),
            "--eval_batch_size",
            "1",
            "--accelerator",
            "cpu",
            "--save_model_dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert "test/loss_epoch" in evaluated.stdout
    assert "test/accuracy_epoch" in evaluated.stdout
    resumed = subprocess.run(
        command
        + [
            "--mode",
            "train",
            "--config_path",
            str(tiny_config),
            "--trainer_ckpt",
            str(tmp_path / "run" / "last.ckpt"),
            "--train_file",
            str(path),
            "--valid_file",
            str(path),
            "--train_batch_size",
            "1",
            "--max_steps",
            "3",
            "--max_epochs",
            "-1",
            "--accelerator",
            "cpu",
            "--save_model_dir",
            str(tmp_path),
            "--save_model_name",
            "resumed",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "max_steps=3" in resumed.stderr
    assert (tmp_path / "resumed" / "model" / "model.safetensors").is_file()
    resumed_state = torch.load(
        tmp_path / "resumed" / "last.ckpt", map_location="cpu", weights_only=False
    )
    assert resumed_state["global_step"] == 3
