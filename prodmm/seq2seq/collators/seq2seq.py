from prolm.unitokenizer.unitokenizer import UniTokenizer


class Seq2SeqCollator:

    def __init__(self, tokenizer: UniTokenizer):
        self.tokenizer = tokenizer

    def collate(self, batch):
        if len(batch) == 0:
            print("Received an empty batch.")
            return None  # or handle the empty batch case appropriately

        input_sequences = [item["S"] for item in batch]
        target_sequences = [item["T"] for item in batch]

        encoder_input_dict = self.tokenizer(
            input_sequences, return_tensors="pt", special_add="encoder"
        )

        decoder_input_dict = self.tokenizer(
            target_sequences, return_tensors="pt", special_add="decoder"
        )

        input_ids = encoder_input_dict["input_ids"]
        attention_mask = encoder_input_dict["attention_mask"]
        labels = decoder_input_dict["input_ids"][
            :, 1:
        ]  # The model will automatically shift the labels and prepend the decoder_input_ids with the decoder_start_token_id
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, batch):
        return self.collate(batch)
