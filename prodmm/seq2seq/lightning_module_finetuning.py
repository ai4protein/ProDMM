from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from prodmm.seqseq.modeling_prolm_seq2seq import ProLMForConditionalGeneration
from prodmm.seq2seq.configuration_prolm import ProLMConfig
from torchmetrics import Accuracy
import torch
from transformers import get_scheduler
from typing import Any
from torch.utils.data import DataLoader
from .dataset import get_dataset
from .collators import Seq2SeqCollator
import numpy as np
import csv
import os
def check_freeze(model):
    for name, param in model.named_parameters():
        # if 'encoder' in name:
        if param.requires_grad:
            print(f"{name} is not frozen")
        else:
            print(f"{name} is frozen")

class LightningProLM(LightningModule):

    def __init__(
        self,
        pretrain_task="mlm",
        config_path='',
        lr=None,
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        warmup_steps=None,
        warmup_max_steps=None,
        weight_dacay=0.01,
        scheduler_type="linear",
        eval_determinstic=False,
        encoder_ckpt=None,
        args=None,
        tokenizer=None,
        cds_gen=None,
        task_type=None,
        save_output_result=None,
    ):
        print(f"Trying to load config from: {config_path}")
        try:
            self.config = ProLMConfig.from_pretrained(config_path)
            print("Config loaded successfully.")
        except Exception as e:
            print(f"Failed to load config: {e}")
        super().__init__()
        self.pretrain_task = pretrain_task
       
        if self.pretrain_task == "mlm":
            self.model = ProLMForMaskedLM(self.config)
        elif self.pretrain_task == "seq2seq":     
            if args.pretrain_ckpt is not None:
                print(f"Load model from the checkpoint{args.pretrain_ckpt}")
                self.model = ProLMForConditionalGeneration.from_pretrained(args.pretrain_ckpt, cds_gen=args.cds_gen)
            else:
                print("Init model from the encoder checkpoint")
                self.model = ProLMForConditionalGeneration(self.config, encoder_ckpt)
            if args.frozen_encoder and not args.use_decoder:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                #check_freeze(self.model)
        self.lr = lr
        self.adam_epsilon = adam_epsilon
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.weight_decay = weight_dacay
        self.scheduler_type = scheduler_type
        self.warmup_max_steps = warmup_max_steps
        self.warmup_steps = warmup_steps
        self.eval_determinstic = eval_determinstic
        self.args = args
        self.validation_outputs = [] 
        self.test_outputs = []
        self.save_output_result = save_output_result
        self.save_hyperparameters()
        if self.args.use_decoder:
                self.eval_collator = DecoderCollator(tokenizer)
        else:
            self.eval_collator = Seq2SeqCollator(tokenizer)
        self._testset = get_dataset(self.args.test_file, streaming=False)
        self.tokenizer = tokenizer
        self.cds_gen = args.cds_gen
        self.task_type = args.task_type
        self.use_decoder = args.use_decoder
    def compute_accuracy(self, preds, labels):
        # Calculate the number of correct predictions
        correct = (preds == labels).float()
        # Calculate accuracy
        accuracy = correct.sum() / len(correct)
        return accuracy

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = self.model(
            input_ids=input_ids, 
            labels=labels, 
            attention_mask=attention_mask
        )
        loss = output.loss
        perplexity = torch.exp(loss.detach())

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
            "train/perplexity",
            perplexity,
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
        print(f"validation_step - Model training mode: {self.model.training}")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = self.model(
            input_ids=input_ids, 
            labels=labels, 
            attention_mask=attention_mask
        )

        loss = output.loss
        preds = output.logits.argmax(dim=-1)
        non_padding_mask = labels != self.config.pad_token_id
        preds = preds[non_padding_mask]
        labels = labels[non_padding_mask]

        acc = self.compute_accuracy(preds[:-1], labels[:-1])
        perplexity = torch.exp(loss.detach()) if not torch.isnan(loss) else torch.tensor(0.0)

        self.validation_outputs.append({'val/loss': loss.detach(), 'val/acc': acc, 'val/perplexity': perplexity})#, 'test/loss': test_loss.detach(), 'test/acc': test_acc, 'test/perplexity': test_perplexity})


    def on_validation_epoch_end(self) -> None:
        print(f"validation_epoch_end - Model training mode: {self.model.training}")
        avg_loss = torch.stack([x['val/loss'] for x in self.validation_outputs]).mean()
        avg_acc = torch.stack([x['val/acc'] for x in self.validation_outputs]).mean()
        avg_perplexity = torch.stack([x['val/perplexity'] for x in self.validation_outputs]).mean()           
        for x in self.validation_outputs:
            print(x['val/acc'])
        
        print('val_acc_meam',avg_acc)
        # Logging epoch metrics
        self.log("val/loss_epoch", avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.trainer.num_devices > 1)
        self.log("val/accuracy_epoch", avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.trainer.num_devices > 1)
        self.log("val/perplexity_epoch", avg_perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.trainer.num_devices > 1)
        self.validation_outputs.clear()

    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        print(f"test_step - Model training mode: {self.model.training}")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss = None
        perplexity = None

        output = self.model(
            input_ids=input_ids, 
            labels=labels, 
            attention_mask=attention_mask
        )

        self.test_outputs.append({
            "outputs": outputs,
            "labels": labels
        })
        
        return None
  
    def on_test_epoch_end(self) -> None:
        print(f"on_test_epoch_end - Model training mode: {self.model.training}")
        
        output_results = []

        test_loss = torch.tensor(1.0)
        test_losses = []
        test_accs = []
        test_perplexities = []

        for test_batch in self.test_dataloader():
            test_batch = {k: v.to(self.device) for k, v in test_batch.items()}
            
            test_input_ids = test_batch["input_ids"]
            test_attention_mask = test_batch["attention_mask"]
            test_labels = test_batch["labels"]

            test_output = self.model(
                input_ids=test_input_ids, 
                labels=test_labels, 
                attention_mask=test_attention_mask
            )
                        
            if test_loss != torch.tensor(0.0):      
                test_loss = test_output.loss
                test_perplexity = torch.exp(test_loss.detach()) if not torch.isnan(test_loss) else torch.tensor(0.0)
                test_preds = test_output.logits.argmax(dim=-1)
            
            test_non_padding_mask = test_labels != self.config.pad_token_id
            test_preds = test_preds[test_non_padding_mask]
            test_labels = test_labels[test_non_padding_mask]
            
            test_acc = self.compute_accuracy(test_preds[:-1], test_labels[:-1])

            test_losses.append(test_loss.detach())
            test_accs.append(test_acc)
            test_perplexities.append(test_perplexity)

        test_avg_loss = torch.stack(test_losses).mean()
        test_avg_acc = torch.stack(test_accs).mean()
        test_avg_perplexity = torch.stack(test_perplexities).mean()
        
        
        # Logging epoch metrics
        self.log("test/loss_epoch", test_avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.trainer.num_devices > 1)
        self.log("test/accuracy_epoch", test_avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.trainer.num_devices > 1)
        self.log("test/perplexity_epoch", test_avg_perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.trainer.num_devices > 1)

        self.test_outputs.clear()

    def test_dataloader(self):
        return DataLoader(
            self._testset,
            collate_fn=self.eval_collator,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )
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
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
