"""Run small, local model checks without downloading pretrained weights."""

import torch

from prodmm.encoder.configuration_prolm import ProLMConfig
from prodmm.encoder.modeling_prolm import ProLMForMaskedLM
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer
from prodmm.seq2seq.configuration_prolm import ProLMConfig as Seq2SeqConfig
from prodmm.seq2seq.modeling_prolm_seq2seq import ProLMForConditionalGeneration
from prodmm.seq2seq_tokenizer.prompttokenizer import PromptTokenizer


def test_encoder_forward():
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
    model = ProLMForMaskedLM(config).eval()
    inputs = tokenizer(["<aas>MAG</aas>", "<cds>ATGGCT</cds>"])
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    assert outputs.logits.shape == (*inputs["input_ids"].shape, len(tokenizer))
    assert torch.isfinite(outputs.logits).all()


def test_seq2seq_model_and_vocabulary_are_available():
    assert issubclass(ProLMForConditionalGeneration, torch.nn.Module)
    tokenizer = PromptTokenizer()
    assert tokenizer.encode("<aas>MAG<aas2cds></aas>") == [
        tokenizer.cls_token_id,
        tokenizer.stoi("M"),
        tokenizer.stoi("A"),
        tokenizer.stoi("G"),
        tokenizer.stoi("<aas2cds>"),
        tokenizer.sep_token_id,
    ]


def test_seq2seq_forward():
    tokenizer = PromptTokenizer()
    config = Seq2SeqConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
    model = ProLMForConditionalGeneration(config).eval()
    inputs = tokenizer(["<aas>MAG<aas2cds></aas>"])
    labels = tokenizer(["<cds>ATGGCTGGT</cds>"], special_add="decoder")["input_ids"][
        :, 1:
    ]
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
    assert outputs.logits.shape == (1, 4, len(tokenizer))
    assert torch.isfinite(outputs.loss)
