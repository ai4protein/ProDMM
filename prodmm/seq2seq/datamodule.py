"""Data loaders for source/target JSONL records."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .collators import Seq2SeqCollator
from .dataset import get_dataset


class FinetuningDataModule(LightningDataModule):
    """Load only the splits required by the requested Lightning stage."""

    def __init__(
        self,
        tokenizer,
        train_file=None,
        valid_file=None,
        test_file=None,
        train_batch_size=4,
        eval_batch_size=4,
        seed=42,
        streaming=False,
        num_workers=0,
    ):
        super().__init__()
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.streaming = streaming
        self.seed = seed
        self.collator = Seq2SeqCollator(tokenizer)

    def setup(self, stage=None):
        if stage in (None, "fit") and self.train_file is not None:
            self._trainset = get_dataset(self.train_file, streaming=self.streaming)
            if self.streaming:
                self._trainset = self._trainset.shuffle(seed=self.seed)
        if stage in (None, "fit", "validate") and self.valid_file is not None:
            self._validset = get_dataset(self.valid_file, streaming=self.streaming)
        if stage in (None, "test") and self.test_file is not None:
            self._testset = get_dataset(self.test_file, streaming=self.streaming)

    def _loader(self, dataset, batch_size, shuffle=False):
        return DataLoader(
            dataset,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            batch_size=batch_size,
            shuffle=shuffle and not self.streaming,
        )

    def train_dataloader(self):
        return self._loader(self._trainset, self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self._loader(self._validset, self.eval_batch_size)

    def test_dataloader(self):
        return self._loader(self._testset, self.eval_batch_size)
