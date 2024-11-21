from torch.utils.data import DataLoader, DistributedSampler
from lightning import LightningDataModule
from .dataset import get_dataset
from .collators import  Seq2SeqCollator


class FinetuningDataModule(LightningDataModule):

    def __init__(
        self,
        tokenizer,
        train_file,
        valid_file,
        test_file,
        train_batch_size,
        eval_batch_size,
        seed,
        streaming=False,
        collator="mlm",  # mlm or seq2seq
        noise_p=0.15,
        to_mask_p=0.8,
        to_random_p=0.1,
        null_token="-",
        num_workers=0,
        args=None,
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.args = args

        if collator == "seq2seq": 
            self.train_collator = Seq2SeqCollator(tokenizer)
            self.eval_collator = Seq2SeqCollator(tokenizer)
        self.streaming = streaming
        self.seed = seed

    def setup(self, stage=None):
        if stage == "fit":
            self._trainset = get_dataset(self.train_file, streaming=False)
            self._validset = get_dataset(self.valid_file, streaming=False)
        elif stage == "test":
            self._testset = get_dataset(self.test_file, streaming=False)
        else:
            pass

    def train_dataloader(self):
        return DataLoader(
            self._trainset,
            collate_fn=self.train_collator,
            num_workers=self.num_workers, 
            shuffle=True,
            batch_size=self.train_batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self._validset,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
            shuffle=False,
            batch_size=self.eval_batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self._testset,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
            shuffle=False,
            batch_size=self.eval_batch_size,
        )
