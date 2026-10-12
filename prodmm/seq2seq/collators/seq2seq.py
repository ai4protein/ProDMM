"""Convert paired tagged sequences into model inputs and target labels."""

from prodmm.seq2seq_tokenizer.prompttokenizer import PromptTokenizer


class Seq2SeqCollator:
    def __init__(self, tokenizer: PromptTokenizer):
        self.tokenizer = tokenizer

    def collate(self, batch):
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        sources = self.tokenizer([item["S"] for item in batch], special_add="encoder")
        targets = self.tokenizer([item["T"] for item in batch], special_add="decoder")

        # Labels omit BOS. The model prepends it when shifting decoder inputs.
        labels = targets["input_ids"][:, 1:].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": sources["input_ids"],
            "attention_mask": sources["attention_mask"],
            "labels": labels,
        }

    def __call__(self, batch):
        return self.collate(batch)
