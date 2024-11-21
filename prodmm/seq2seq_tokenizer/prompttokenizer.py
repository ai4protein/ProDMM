from typing import List, Union, Optional, Tuple, Any, Literal
import torch
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AAS_TOKENS = list("LAGVSERTIDPKQNFYMHWC*")
NCDS_TOKENS = list("acgt")
CDS_TOKENS = [
    "aaa",
    "aat",
    "aac",
    "aag",
    "ata",
    "att",
    "atc",
    "atg",
    "aca",
    "act",
    "acc",
    "acg",
    "aga",
    "agt",
    "agc",
    "agg",
    "taa",
    "tat",
    "tac",
    "tag",
    "tta",
    "ttt",
    "ttc",
    "ttg",
    "tca",
    "tct",
    "tcc",
    "tcg",
    "tga",
    "tgt",
    "tgc",
    "tgg",
    "caa",
    "cat",
    "cac",
    "cag",
    "cta",
    "ctt",
    "ctc",
    "ctg",
    "cca",
    "cct",
    "ccc",
    "ccg",
    "cga",
    "cgt",
    "cgc",
    "cgg",
    "gaa",
    "gat",
    "gac",
    "gag",
    "gta",
    "gtt",
    "gtc",
    "gtg",
    "gca",
    "gct",
    "gcc",
    "gcg",
    "gga",
    "ggt",
    "ggc",
    "ggg",
]

SPECIAL_TOKENS = [
    "<mask>", 
    "<unk>", 
    "<pad>", 
    "<cls>", 
    "<sep>",
    "<bos>",
    "<eos>",
    "-",
]
PROMPT_TOKENS = [
    "<aas2cds>",
    "<cds2pre-ncds>",
    "<cds2post-ncds>",
]
# <mask> : mask token
# <unk> : unknown token
# <pad> : padding token
# <cls> : classification token (for classification tasks)
# <sep> : separation token (for sperating sequences)
# <bos> : begin of sequence token (for generating sequences)
# <eos> : end of sequence token (for generating sequences)
# - : null token
SEP_TOKEN = "<sep>"


class AASTokenizer:

    def __init__(self):
        self.pattern = "|".join(map(re.escape, SPECIAL_TOKENS + PROMPT_TOKENS)) + "|."

    def tokenize(self, text: str) -> List[str]:
        matches = re.findall(self.pattern, text)
        return matches


class CDSTokenizer:

    def __init__(self):
        self.pattern = "|".join(map(re.escape, SPECIAL_TOKENS + PROMPT_TOKENS)) + "|."

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        segments = re.findall(self.pattern, text)
        if len(segments[-1]) > 3:
            sequence_to_process = segments[:-1]
        else:
            sequence_to_process = segments
        tokens = []
        for i in range(0, len(sequence_to_process), 3):
            new_tokens = ''.join(sequence_to_process[i:i+3])
            tokens.append(new_tokens)
        if len(segments[-1]) > 3:
            tokens.append(segments[-1])
        return tokens


class NCDSTokenizer:

    def __init__(self):
        self.pattern = "|".join(map(re.escape, SPECIAL_TOKENS + PROMPT_TOKENS)) + "|."

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        matches = re.findall(self.pattern, text)
        return matches


class TaggedTokneizer:

    def __init__(self):
        self.legal_data_types = {"ncds", "cds", "aas","pre-ncds","post-ncds","aas2cds","cds2pre-ncds","cds2post-ncds"}
        self.partten = r"<(ncds|cds|aas|aas2cds|cds2pre-ncds|cds2post-ncds)>(.*?)</\1>"
        #self.partten = r"<(ncds|cds|aas|aas2cds|cds2pre-ncds|cds2post-ncds)>(.*?)</\1>|<(aas2cds|cds2pre-ncds|cds2post-ncds)>"
        self.default_data_type = "aas"
        self.aas_tokenizer = AASTokenizer()
        self.ncds_tokenizer = NCDSTokenizer()
        self.cds_tokenizer = CDSTokenizer()

    def tokenize(self, text: Union[str, Tuple[str, str], List[str]]) -> List[str]:        
        has_input_data_type = False
        if isinstance(text, tuple):
            text, data_type = text
            has_input_data_type = True
            assert data_type in self.legal_data_types, f"Invalid data_type: {data_type}"
            print('isinstance(text, tuple)')
        elif isinstance(text, list):
            text, data_type = text
            has_input_data_type = True
            assert data_type in self.legal_data_types, f"Invalid data_type: {data_type}"
            print('isinstance(text, list)')
        else:
            data_type = None
        matches = re.findall(self.partten, text)
        if len(matches) == 0:
            if data_type is None:
                data_type = self.default_data_type
            text = f"<{data_type}>{text}</{data_type}>"
            matches = re.findall(self.partten, text)
        else:
            data_type = None
            if has_input_data_type:
                print("Warning: data_type in text is ignored.")
        #print(matches,"matches")
        tokens = []
        for tag, content in matches:
            if tag == "ncds":
                tokens += self.ncds_tokenizer.tokenize(content)
                tokens += [SEP_TOKEN, ]
            elif tag == "cds":
                tokens += self.cds_tokenizer.tokenize(content)
                tokens += [SEP_TOKEN, ]
            elif tag == "aas":
                tokens += self.aas_tokenizer.tokenize(content)
                tokens += [SEP_TOKEN, ]
            else:
                raise ValueError(f"Invalid tag: {tag}")
        return tokens[:-1] # Remove the last SEP_TOKEN


class PromptTokenizer:

    def __init__(
        self,
    ):
        self.tokenizer = TaggedTokneizer()
        self.vocab_file = BASE_DIR / "prompt_vocab.txt"
        self.vocab = self.load_vocab()
        print(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.special_tokens = SPECIAL_TOKENS
        self.special_tokens = {t: self.vocab[t] for t in self.special_tokens}
        
        self.cls_token = "<cls>"
        self.sep_token = "<sep>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.unk_token = "<unk>"
        self.null_token = "-"
        
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.mask_token_id = self.vocab[self.mask_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.null_token_id = self.vocab[self.null_token]
        
        self.aas_tokens = AAS_TOKENS
        self.ncds_tokens = NCDS_TOKENS
        self.cds_tokens = CDS_TOKENS
        self.prompt_tokens = PROMPT_TOKENS
        self.all_tokens = list(self.vocab.keys())
        
        assert len(self.all_tokens) == len(AAS_TOKENS) + len(NCDS_TOKENS) + len(CDS_TOKENS) + len(SPECIAL_TOKENS) + len(PROMPT_TOKENS)
        
    def load_vocab(self) -> List[str]:
        with open(self.vocab_file, "r") as f:
            vocab = f.read().splitlines()
        vocab = {v: i for i, v in enumerate(vocab)}
        return vocab

    def itos(self, token_id: int) -> str:
        token_id = int(token_id)
        return self.inverse_vocab[token_id]

    def stoi(self, token: str) -> int:
        return self.vocab[token] if token in self.vocab else self.unk_token_id

    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        return [self.cls_token] + tokens + [self.sep_token]

    def tokenize(
        self,
        text: Union[str, Tuple[str, str], List[str]],
        max_length: int = None,
        special_add = "encoder", # encoder, decoder, none
    ) -> List[str]:
        
        tokens = self.tokenizer.tokenize(text)
        if max_length:
            tokens = tokens[:max_length]
        if special_add == "encoder":
            tokens = [self.cls_token] + tokens + [self.sep_token]
        elif special_add == "decoder":
            tokens = [self.bos_token] + tokens + [self.eos_token]
        else:
            pass
        return tokens

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.vocab)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.stoi(t) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # for i in ids:
        #     breakpoint()
        return [self.itos(i) for i in ids]
        #return [self.itos(i) for i in ids if isinstance(i, (int, str))]

    def convert_batch_tokens(self, 
        batch_tokens: List[List[str]], 
        has_special_tokens = False, # whether the batch_tokens has special tokens
        special_add = "encoder",
        padding: bool = True,
        return_attention_mask: bool = True,
        return_length: bool = True,
        return_special_tokens_mask: bool = True,
        return_tensors: Optional[Literal["pt", None]] = None,
    ) -> torch.Tensor:
        if not has_special_tokens:
            if special_add == "encoder":
                batch_tokens = [
                    [self.cls_token] + tokens + [self.sep_token]
                    for tokens in batch_tokens
                ]
            elif special_add == "decoder":
                batch_tokens = [
                    [self.bos_token] + tokens + [self.eos_token]
                    for tokens in batch_tokens
                ]
            else:
                pass
        
        speical_tokens_mask: List[List[int]] = []
        batch_lengths: List[int] = []
        attention_mask: List[List[int]] = []
        #print(batch_tokens)
        for tokens in batch_tokens:
            batch_lengths.append(len(tokens))
            speical_tokens_mask.append(
                [
                    1 if t in self.special_tokens or t not in self.vocab else 0
                    for t in tokens
                ]
            )
            attention_mask.append([1] * len(tokens))
        
        batch_max_length = max(batch_lengths)
        
        if padding:
            pad_token_id = self.pad_token_id
            batch_token_ids = [
                self.convert_tokens_to_ids(tokens)
                + [pad_token_id] * (batch_max_length - len(tokens))
                for tokens in batch_tokens
            ]
            attention_mask = [
                mask + [0] * (batch_max_length - len(mask)) for mask in attention_mask
            ]
            speical_tokens_mask = [
                mask + [0] * (batch_max_length - len(mask))
                for mask in speical_tokens_mask
            ]
            
            if return_tensors == "pt":
                return_results = {}
                return_results["input_ids"] = torch.tensor(
                    batch_token_ids, dtype=torch.long
                )
                if return_attention_mask:
                    return_results["attention_mask"] = torch.tensor(
                        attention_mask, dtype=torch.long
                    )
                if return_length:
                    return_results["length"] = torch.tensor(
                        batch_lengths, dtype=torch.long
                    )
                if return_special_tokens_mask:
                    return_results["special_tokens_mask"] = torch.tensor(
                        speical_tokens_mask, dtype=torch.long
                    )
                return return_results
            else:
                return_results = {"input_ids": batch_token_ids}
                if return_attention_mask:
                    return_results["attention_mask"] = attention_mask
                if return_length:
                    return_results["length"] = batch_lengths
                if return_special_tokens_mask:
                    return_results["special_tokens_mask"] = speical_tokens_mask
                return return_results

        else:
            batch_token_ids = [
                self.convert_tokens_to_ids(tokens) for tokens in batch_tokens
            ]
            return_results = {"input_ids": batch_token_ids}
            if return_attention_mask:
                return_results["attention_mask"] = attention_mask
            if return_length:
                return_results["length"] = batch_lengths
            if return_special_tokens_mask:
                return_results["special_tokens_mask"] = speical_tokens_mask
            return return_results
        
    def encode(
        self,
        text: Union[str, Tuple[str, str], List[str]],
        max_length: int = None,
        special_add: str = "encoder",
    ) -> torch.Tensor:
        tokens = self.tokenize(
            text,
            special_add=special_add,
            max_length=max_length,
        )
        return self.convert_tokens_to_ids(tokens)

    def batch_encode(
        self,
        batch_text: Union[List[str], List[Tuple[str, str]], List[List[str]]],
        special_add: str = "encoder",
        max_length: Optional[int] = None,
        padding: bool = True,
        return_attention_mask: bool = True,
        return_length: bool = False,
        return_special_tokens_mask: bool = False,
        return_tensors=None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Any]:
        batch_tokens: List[List[str]] = []        
        for text in batch_text:
            tokens = self.tokenize(
                text,
                special_add=None,
                max_length=max_length,
            )
            batch_tokens.append(tokens)
        return self.convert_batch_tokens(
            batch_tokens = batch_tokens, 
            has_special_tokens = False,
            special_add = special_add,
            padding = padding,
            return_attention_mask = return_attention_mask,
            return_length = return_length,
            return_special_tokens_mask = return_special_tokens_mask,
            return_tensors = return_tensors,
        )

    def __call__(
        self,
        batch_text: Union[List[str], List[Tuple[str, str]], List[List[str]]],
        special_add: str = "encoder",
        max_length: Optional[int] = None,
        padding: bool = True,
        return_attention_mask: bool = True,
        return_length: bool = True,
        return_special_tokens_mask: bool = True,
        return_tensors = "pt",
    ):
        return self.batch_encode(
            batch_text,
            special_add=special_add,
            max_length=max_length,
            padding=padding,
            return_attention_mask=return_attention_mask,
            return_length=return_length,
            return_special_tokens_mask=return_special_tokens_mask,
            return_tensors=return_tensors,
        )
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
    ) -> str:
        # Convert token_ids to a list if it's a tensor
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens = self.convert_ids_to_tokens(token_ids)
        # Remove special tokens if needed
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        # Reconstruct text from tokens
        text = ''.join(tokens)
        return text