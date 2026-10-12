import pytest

from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer
from prodmm.seq2seq_tokenizer.prompttokenizer import PromptTokenizer


def test_multimodal_encoding_and_padding():
    tokenizer = UniTokenizer()
    result = tokenizer(["<aas>MA</aas><cds>ATGGCT</cds>", "<ncds>AC</ncds>"])
    assert tokenizer.convert_ids_to_tokens(result["input_ids"][0].tolist()) == [
        "<cls>",
        "M",
        "A",
        "<sep>",
        "atg",
        "gct",
        "<sep>",
    ]
    assert result["attention_mask"][1].tolist() == [1, 1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ("", []),
        ("ATGGCT", ["atg", "gct"]),
        ("ATGGCT<cds2pre-ncds>", ["atg", "gct", "<cds2pre-ncds>"]),
        ("<mask>ATG<mask>GCT", ["<mask>", "atg", "<mask>", "gct"]),
        ("<cds2post-ncds>ATG", ["<cds2post-ncds>", "atg"]),
    ],
)
def test_prompt_cds_preserves_special_tokens(sequence, expected):
    tokenizer = PromptTokenizer()
    assert tokenizer.tokenize(f"<cds>{sequence}</cds>", special_add=None) == expected


def test_prompt_cds_rejects_partial_codons():
    tokenizer = PromptTokenizer()
    with pytest.raises(ValueError, match="multiple of 3"):
        tokenizer.tokenize("<cds>ATGA</cds>")


def test_prompt_tokenizer_initialization_is_quiet(capsys):
    PromptTokenizer()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("tokenizer_class", [UniTokenizer, PromptTokenizer])
def test_tensor_ids_decode_without_manual_list_conversion(tokenizer_class):
    tokenizer = tokenizer_class()
    inputs = tokenizer(["<aas>MA</aas>"])
    assert tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) == [
        "<cls>",
        "M",
        "A",
        "<sep>",
    ]
