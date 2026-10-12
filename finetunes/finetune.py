import argparse
import numpy as np
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import EarlyStopping
from Bio.Seq import Seq
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import os
import torch
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer
from prodmm.encoder.configuration_prolm import ProLMConfig
from finetunes.ft_model import ProBLAMForSequenceClassification
from transformers import get_scheduler
from lightning import LightningDataModule, LightningModule, seed_everything, Trainer
from torch.utils.data import DataLoader
import json
from sklearn.model_selection import KFold
from datasets import load_dataset, Features, Value, Sequence


def get_logger(logger_type, run_name=None, project=None, entity=None, log_args={}):
    if logger_type == "wandb":
        if entity is None:
            raise ValueError("entity must be provided when using wandb logger")
        logger = WandbLogger(
            name=run_name,
            project=project,
            entity=entity,
            config=vars(log_args),
        )
    elif logger_type == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=f"tb_logs/{run_name}",
            name=run_name,
        )
    else:
        logger = None
    return logger

def get_strategy(strategy, find_unused_parameters=True):
    if strategy == "ddp":
        return DDPStrategy(find_unused_parameters=find_unused_parameters)
    else:
        return strategy

def strtobool(val):
    """Convert a string representation of truth to True or False."""
    val_lower = val.lower()
    if val_lower in ("yes", "y", "true", "t", "1"):
        return True
    elif val_lower in ("no", "n", "false", "f", "0"):
        return False
    else:
        raise ValueError(f"Invalid truth value: {val}")

def get_dataset_finetune(
        file,
        streaming=True,
        problem_type='regression',
):
    dataset = load_dataset(
        "json",
        data_files=[
            file,
        ],
        features=Features(
            {
                "S": Value("string"),
                "label": Sequence(Value("int32")) if problem_type == "multi_label_classification" else Value("int32"),
                "value": Value("float32"),
                "L": Value("int32"),
            }
        ),
        split="train",
        streaming=streaming,
    )
    return dataset


AVAIL_GPUS = min(1, torch.cuda.device_count())


class BertCollatorFinetune:

    def __init__(
            self,
            tokenizer: UniTokenizer,
            null_token="-",
            problem_type="regression",
            amino=False,
    ):
        self.tokenizer = tokenizer
        self.null_token = null_token
        self.null_token_id = tokenizer.get_vocab()[null_token]
        self.problem_type = problem_type
        self.amino = amino

    def collate(self, batch):
        # seqs = [['<ncds>'+ item["S"] + '</ncds>'] for item in batch]

        if self.problem_type == "multi_label_classification":
            seqs = [(item["S"], 'aas') for item in batch]
        else:
            if self.amino:
                seqs = []
                for item in batch:
                    dna_seq = Seq(item["S"])
                    protein_seq = dna_seq.translate()
                    seqs.append((str(protein_seq), 'aas'))
            else:
                seqs = [(item["S"], 'cds') for item in batch]

        seqs = [self.tokenizer.tokenize(text=item, special_add=None) for item in seqs]

        if self.problem_type == "regression":
            labels = torch.tensor([item["value"] for item in batch])
        else:
            labels = torch.tensor([item["label"] for item in batch])

        input_dict = self.tokenizer.convert_batch_tokens(
            seqs, special_add="encoder", return_tensors="pt")

        input_ids = input_dict["input_ids"]
        attention_mask = input_dict["attention_mask"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, batch):
        return self.collate(batch)


class ProLMFinetune(LightningModule):

    def __init__(
            self,
            model,
            config=None,
            tokenizer=None,
            lr=None,
            adam_epsilon=1e-8,
            adam_beta1=0.9,
            adam_beta2=0.999,
            warmup_steps=None,
            warmup_max_steps=None,
            weight_dacay=0.01,
            scheduler_type="linear",
            eval_determinstic=False,
            args=None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.adam_epsilon = adam_epsilon
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.weight_decay = weight_dacay
        self.scheduler_type = scheduler_type
        self.warmup_max_steps = warmup_max_steps
        self.warmup_steps = warmup_steps
        self.eval_determinstic = eval_determinstic
        self.tokenizer = tokenizer
        self.args = args
        self.save_hyperparameters(ignore=["config", "tokenizer", "model"])

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = self.model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )
        loss = output.loss

        self.log(
            "train/loss",
            loss.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=self.trainer.num_devices > 1,
        )
        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            rank_zero_only=True,
        )
        self.log(
            "train/step",
            self.global_step,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            rank_zero_only=True,
        )
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        if self.eval_determinstic:
            input_ids[labels > 0] = labels[labels > 0]
            labels = input_ids.clone()
        output = self.model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )
        loss = output.loss
        self.log(
            "val_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.trainer.num_devices > 1,
        )
        return loss

    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        if self.eval_determinstic:
            input_ids[labels > 0] = labels[labels > 0]
            labels = input_ids.clone()
        output = self.model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )
        loss = output.loss
        self.log(
            "test/loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.trainer.num_devices > 1,
        )
        return loss

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        if self.eval_determinstic:
            input_ids[labels > 0] = labels[labels > 0]
            labels = input_ids.clone()
        output = self.model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )
        logits = output.logits
        return logits

    def configure_optimizers(self) -> Any:
        no_decay = [
            "bias",
            "LayerNorm.weight",
        ]  # no decay for bias and LayerNorm.weight

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            eps=self.adam_epsilon,
            betas=(self.adam_beta1, self.adam_beta2),
        )
        scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.warmup_max_steps,
        )
        scheduler = {"scheduler": scheduler,
                     "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


class LanguageDataModuleFintune(LightningDataModule):

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
            null_token="-",
            num_workers=0,
            problem_type="regression",
            amino=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.streaming = streaming
        self.seed = seed
        self.problem_type = problem_type
        self.train_collator = BertCollatorFinetune(
            tokenizer,
            null_token=null_token,
            problem_type=self.problem_type,
            amino=amino,
        )
        self.eval_collator = BertCollatorFinetune(
            tokenizer,
            null_token=null_token,
            problem_type=self.problem_type,
            amino=amino,
        )

    def setup(self, stage=None):
        if stage == "fit":
            self._trainset = get_dataset_finetune(
                self.train_file, 
                streaming=self.streaming,
                problem_type=self.problem_type
            )
            self._validset = get_dataset_finetune(
                self.valid_file, 
                streaming=False, 
                problem_type=self.problem_type
            )
        elif stage == "test":
            self._testset = get_dataset_finetune(
                self.test_file, 
                streaming=False, 
                problem_type=self.problem_type
            )
        elif stage == "predict":
            self._predictset = get_dataset_finetune(
                self.test_file, 
                streaming=False, 
                problem_type=self.problem_type
            )
        else:
            pass

    def train_dataloader(self):
        return DataLoader(
            self._trainset,
            collate_fn=self.train_collator,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._validset,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self._testset,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self._predictset,
            collate_fn=self.eval_collator,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_gradient_norm",
        type=strtobool,
        default="False",
        help="Watch model gradients",
        required=False,
    )

    parser.add_argument("--tokenizer_path", type=str,
                        default="AI4Protein/prodmm_encoder")
    parser.add_argument("--config_path", type=str,
                        default="AI4Protein/prodmm_encoder")

    # Parameters for data
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)

    # Parameters for training
    parser.add_argument("--accumulate_grad_batches", default=16, type=int)
    parser.add_argument("--global_train_batch_size", default=None, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--adam_epsilon", default=1e-07, type=float)
    parser.add_argument("--max_steps", default=1000000, type=int)
    parser.add_argument("--warmup_max_steps", default=None, type=int)
    parser.add_argument("--max_epochs", default=1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--gradient_clip_value", default=1.0, type=float)
    parser.add_argument("--gradient_clip_algorithm", default="norm", type=str)
    parser.add_argument("--precision", default="bf16-mixed", type=str)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--scheduler_type", default="linear", type=str)
    parser.add_argument("--load_from", default="", type=str)

    # Parameters for fine-tuning
    parser.add_argument("--problem_type", default="regression", type=str)
    parser.add_argument("--num_labels", default=5, type=int)
    parser.add_argument("--cross_fold", default=0, type=int)
    parser.add_argument("--cross_fold_index", default=0, type=int)
    parser.add_argument("--task_name", default='Localization', type=str)
    parser.add_argument("--amino", default=False, type=strtobool)

    # Parameters for validation and checkpoint
    parser.add_argument(
        "--check_val_every_n_epoch", default=None, type=int
    )  # val per epoch
    parser.add_argument(
        "--val_check_interval", default="1.0", type=str
    )  # val per batch
    parser.add_argument("--save_model_dir", default="checkpoint", type=str)
    parser.add_argument("--save_model_name", required=True, type=str)
    parser.add_argument("--save_interval", default=None, type=int)

    # Parameters for logging
    parser.add_argument("--log_steps", default=5, type=int)
    parser.add_argument("--logger", default="tensorboard", type=str)
    parser.add_argument("--logger_project", default="openesm", type=str)
    parser.add_argument("--logger_run_name", default="openesm", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)

    # parameters for distributed training
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--strategy", default="auto", type=str)

    # parameters for resume training
    parser.add_argument("--trainer_ckpt", default=None, type=str)
    args = parser.parse_args()

    # set save_model_dir
    args.save_model_dir = args.save_model_dir + "/" + \
        args.task_name + '/' + str(args.cross_fold_index)

    # val_check_interval
    if "." in args.val_check_interval:
        args.val_check_interval = float(args.val_check_interval)
    else:
        args.val_check_interval = int(args.val_check_interval)

    if args.max_epochs is None or args.max_epochs < 0:
        args.max_epochs = 1000000
    if args.max_steps is None and args.max_epochs is None:
        raise ValueError(
            "max_steps and max_epochs can not be None at the same time")

    if args.warmup_max_steps is None or (args.warmup_max_steps < args.max_steps):
        print("warmup_max_steps should be larger than max_steps")
        print("Set warmup_max_steps to max_steps")
        args.warmup_max_steps = args.max_steps

    return args


def get_trainer(args):
    logger = get_logger(
        logger_type=args.logger,
        run_name=args.logger_run_name,
        project=args.logger_project,
        entity=args.wandb_entity,
        log_args=args,
    )

    strategy = get_strategy(args.strategy)

    if args.global_train_batch_size is not None:
        assert (
            args.accumulate_grad_batches == 1
        ), "accumulate_grad_batches should be 1 when global_train_batch_size is set"
        args.accumulate_grad_batches = args.global_train_batch_size // (
            args.train_batch_size * args.nodes * args.devices
        )
        assert (
            args.global_train_batch_size
            % (args.train_batch_size * args.nodes * args.devices)
            == 0
        ), "global_train_batch_size should be divisible by train_batch_size * nodes * devices"
        assert args.accumulate_grad_batches > 0, "accumulate_grad_batches should be > 0"

        print(
            f"accumulate_grad_batches is set to: {args.accumulate_grad_batches}")

    else:
        args.global_train_batch_size = (
            args.train_batch_size
            * args.nodes
            * args.devices
            * args.accumulate_grad_batches
        )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          dirpath=args.save_model_dir,
                                          filename='model-{epoch:02d}-{val_loss:.2f}',
                                          save_top_k=1, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min')
    trainer = Trainer(
        strategy=strategy,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[early_stopping, checkpoint_callback],
        gradient_clip_val=args.gradient_clip_value,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        logger=logger,
        precision=args.precision,
        num_nodes=args.nodes,
        use_distributed_sampler=False,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        # limit_train_batches=10,
        # limit_val_batches=5
    )

    return trainer


def count_f1_mean(pred, target):
    """
    F1 score with the optimal threshold, Copied from TorchDrug.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """

    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = (
        order
        + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
        * order.shape[1]
    )
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / \
        (all_precision + all_recall + 1e-10)
    return all_f1.mean()


def calculate_metric(prediction, config, args):
    predictset = get_dataset_finetune(
        args.test_file, problem_type=args.problem_type)
    if config.problem_type == "regression":
        labels = np.array([item['value']
                           for item in predictset])
        from scipy.stats import pearsonr
        if args.num_labels == 1:
            # caclualate the pearson correlation coefficient for sample
            prediction = prediction.squeeze()
            metric = pearsonr(prediction, labels)[0]
            # caclualate R^2 for sample
            r_squared = metric ** 2
        else:
            prediction = prediction
            metric = pearsonr(prediction, labels)[0]
            r_squared = metric ** 2
    elif config.problem_type == "single_label_classification":
        labels = np.array([item['label']
                           for item in predictset])
        # calcluate the weighted f1 score
        from sklearn.metrics import f1_score
        prediction = np.argmax(prediction, axis=1)
        metric = f1_score(labels, prediction,
                          average="weighted")
    elif config.problem_type == "multi_label_classification":
        labels = np.array([item['label']
                           for item in predictset])
        # calcluate the weighted f1 score
        from sklearn.metrics import f1_score
        metric = count_f1_mean(
            labels, prediction.to(dtype=torch.float32)).item()

    return metric, r_squared


def main():
    args = parse_args()
    config = ProLMConfig.from_pretrained(args.config_path)

    # Assign the task type to the config
    config.problem_type = args.problem_type

    # Assign the task type to the config
    config.num_labels = args.num_labels

    tokenizer = UniTokenizer()

    config.vocab_size = len(tokenizer)
    print(f"Vocab size: {config.vocab_size}")

    if args.seed is not None:
        seed_everything(args.seed)

    # extract file directory
    args.output_dir = args.train_file[:args.train_file.rfind('/')]
    print(
        f"Fold {args.cross_fold_index}: output directory: {args.output_dir}")

    # if cross_fold_index is 0, then split the data into k folds
    if args.cross_fold_index == 0:
        assert os.path.exists(
            args.train_file), f"Training file {args.train_file} does not exist."

        # open Jsonl file
        with open(args.train_file, "r") as f:
            data = [json.loads(line) for line in f]

        # read data's item['L'] and judge whether greater than 2048, if greater than 2048, then cutoff item['S'] to 2048
        for item in data:
            if item['L'] > 2046:
                item['S'] = item['S'][:2046]
                item['L'] = 2046

        kf = KFold(n_splits=args.cross_fold,
                   shuffle=True, random_state=args.seed)
        for fold, (train_index, test_index) in enumerate(kf.split(data)):
            train_data = [data[i] for i in train_index]
            test_data = [data[i] for i in test_index]

            if not os.path.exists(f"{args.output_dir}/fold_{fold}"):
                os.makedirs(f"{args.output_dir}/fold_{fold}")

            args.train_file = f"{args.output_dir}/fold_{fold}/train.jsonl"
            args.valid_file = f"{args.output_dir}/fold_{fold}/train.jsonl"
            args.test_file = f"{args.output_dir}/fold_{fold}/test.jsonl"
            with open(args.train_file, 'w', encoding='utf-8') as f:
                for temp_dict in train_data:
                    f.write(json.dumps(temp_dict) + '\n')
            with open(args.test_file, 'w', encoding='utf-8') as f:
                for temp_dict in test_data:
                    f.write(json.dumps(temp_dict) + '\n')

            print(
                f"Fold {fold}: train file: {args.train_file}, test file: {args.test_file}")

            del train_data, test_data, temp_dict

    # set logger_run_name
    args.train_file = f"{args.output_dir}/fold_{args.cross_fold_index}/train.jsonl"
    args.valid_file = f"{args.output_dir}/fold_{args.cross_fold_index}/train.jsonl"
    args.test_file = f"{args.output_dir}/fold_{args.cross_fold_index}/test.jsonl"

    if args.amino:
        args.logger_run_name = f"Amino_{args.task_name}_fold_{args.cross_fold_index}"
    else:
        args.logger_run_name = f"{args.task_name}_fold_{args.cross_fold_index}"

    # set trainer
    trainer = get_trainer(args)

    datamodule = LanguageDataModuleFintune(
        tokenizer=tokenizer,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        streaming=False,
        num_workers=args.num_workers,
        problem_type=args.problem_type,
        amino=args.amino,
    )

    model = ProLMFinetune(
        model=ProBLAMForSequenceClassification(
            config, config_path=args.config_path, pooling_head="attention1d"),
        config=config,
        tokenizer=tokenizer,
        lr=args.lr,
        adam_epsilon=args.adam_epsilon,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_steps=args.warmup_steps,
        warmup_max_steps=args.warmup_max_steps,
        weight_dacay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        eval_determinstic=False,
    )

    for param in model.model.prolm.parameters():
        param.requires_grad = False
    # # print parameters, note model.model.embeddings.word_embeddings.parameters() is a generator
    # print(args.config_path)
    # print("Parameters:")
    # for param in model.model.embeddings.word_embeddings.parameters():
    #     print(param)
    #     break

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.trainer_ckpt)

    # predict the test data
    with torch.no_grad():
        output = trainer.predict(model, datamodule=datamodule)
        prediction = torch.cat(output, dim=0)[:, 0].detach().cpu().to(
            dtype=torch.float32).numpy()
        # get labels from datamodule
        metric, r_square = calculate_metric(
            prediction=prediction, config=config, args=args)

        # log the metric
        if args.amino:
            trainer.logger.log_metrics({f"Amino {args.task_name} prediction fold {args.cross_fold_index}/metric": metric,
                                        f"Amino {args.task_name} prediction fold {args.cross_fold_index}/r_square": r_square})
        else:
            trainer.logger.log_metrics({f"{args.task_name} prediction fold {args.cross_fold_index}/metric": metric,
                                        f"{args.task_name} prediction fold {args.cross_fold_index}/r_square": r_square})

    print(f"Fold {args.cross_fold_index} finished")


if __name__ == "__main__":
    main()
