"""Lightning integration for paired-sequence training and evaluation."""

from pathlib import Path

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from transformers import get_scheduler

from .configuration_prolm import ProLMConfig
from .modeling_prolm_seq2seq import ProLMForConditionalGeneration


class Seq2SeqModule(LightningModule):
    """Wrap the seq2seq model without owning datasets or data loaders.

    Evaluation uses teacher-forced logits. Loss and accuracy are averaged
    over target tokens, excluding labels masked with -100 by the collator.
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint: str | Path | None = None,
        encoder_checkpoint: str | Path | None = None,
        frozen_encoder: bool = False,
        cds_gen: bool = False,
        task_type: str = "prot_cds",
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        config = ProLMConfig.from_pretrained(config_path)
        if checkpoint is None:
            self.model = ProLMForConditionalGeneration(
                config, encoder_checkpoint, cds_gen=cds_gen, task_type=task_type
            )
        else:
            self.model = ProLMForConditionalGeneration.from_pretrained(
                checkpoint, cds_gen=cds_gen, task_type=task_type
            )
        if frozen_encoder:
            self.model.encoder.requires_grad_(False)

        self.validation_loss = MeanMetric()
        self.validation_accuracy = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_accuracy = MeanMetric()

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def _evaluation_step(self, batch, loss_metric, accuracy_metric):
        output = self(**batch)
        labels = batch["labels"]
        valid = labels != -100
        token_count = valid.sum()
        if token_count > 0:
            predictions = output.logits.argmax(dim=-1)
            accuracy = (predictions[valid] == labels[valid]).float().mean()
            loss_metric.update(output.loss.detach(), weight=token_count)
            accuracy_metric.update(accuracy, weight=token_count)

    def validation_step(self, batch, batch_idx):
        self._evaluation_step(batch, self.validation_loss, self.validation_accuracy)

    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch, self.test_loss, self.test_accuracy)

    def _log_epoch_metrics(self, stage, loss_metric, accuracy_metric):
        # MeanMetric synchronizes its accumulated values across processes.
        loss = loss_metric.compute()
        accuracy = accuracy_metric.compute()
        self.log(f"{stage}/loss_epoch", loss, prog_bar=True)
        self.log(f"{stage}/accuracy_epoch", accuracy, prog_bar=True)
        self.log(f"{stage}/perplexity_epoch", loss.exp())
        loss_metric.reset()
        accuracy_metric.reset()

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val", self.validation_loss, self.validation_accuracy)

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test", self.test_loss, self.test_accuracy)

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                group = (
                    no_decay if parameter.ndim < 2 or name.endswith("bias") else decay
                )
                group.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_epsilon,
        )
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# Preserve the historical import name. Constructor options are documented
# by Seq2SeqModule rather than inferred from an argparse Namespace.
LightningProLM = Seq2SeqModule
