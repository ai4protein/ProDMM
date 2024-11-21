import argparse
from lightning import seed_everything,Callback
from lightning.pytorch.trainer import Trainer
from prodmm.seq2seq.lightning_module_finetuning import LightningProLM
from prodmm.seq2seq.datamodule import PreTrainingDataModule, FinetuningDataModule
from prodmm.seq2seq.train_utils import (
    get_logger,
    get_callbacks,
    get_finetuning_callbacks,
    get_strategy,
    init_train,
    strtobool,
)
from prodmm.unitokenizer.prompttokenizer import PromptTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
init_train()
class MonitorPatienceCallback(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Access the early stopping callback
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                # Print the current patience (wait_count) after each validation step
                print(f"Current patience count: {callback.wait_count}/{callback.patience}")
def check_freeze(model):
    for name, param in model.named_parameters():
        if 'encoder' in name:
            if param.requires_grad:
                print(f"{name} is not frozen")
            else:
                print(f"{name} is frozen")

class TestCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        print("\nRunning Test...")
        trainer.test(pl_module, dataloaders=pl_module.test_dataloader())
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_gradient_norm",
        type=strtobool,
        default="False",
        help="Watch model gradients",
        required=False,
    )

    parser.add_argument("--tokenizer_path", type=str, default="AI4Protein/prolm_8M")
    parser.add_argument("--config_path", type=str, default="AI4Protein/prolm_8M")

    # Parameters for data
    parser.add_argument("--pretrain_task", type=str, choices=["seq2seq", "mlm", "seq2mlm"])
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--train_batch_size", default=2048, type=int)
    parser.add_argument("--eval_batch_size", default=2048, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)

    # Parameters for training
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--global_train_batch_size", default=None, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--adam_epsilon", default=1e-07, type=float)
    parser.add_argument("--max_steps", default=1000000, type=int)
    parser.add_argument("--warmup_max_steps", default=None, type=int)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--gradient_clip_value", default=1.0, type=float)
    parser.add_argument("--gradient_clip_algorithm", default="norm", type=str)
    parser.add_argument("--precision", default="bf16-mixed", type=str)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--scheduler_type", default="linear", type=str)

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
    parser.add_argument("--patience", default=None, type=int)
    parser.add_argument("--cds_gen", default=False, type=bool)
    parser.add_argument("--task_type", default='prot_cds', type=str, choices=["prot_cds", "cds_promoter", "promoter_cds"])
    parser.add_argument("--use_decoder", default=False, type=bool)
    parser.add_argument("--save_output_result", type=str, default=None)
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
    parser.add_argument("--encoder_ckpt", default=None, type=str)
    parser.add_argument("--pretrain_ckpt", default=None, type=str)
    parser.add_argument("--frozen_encoder", default=False, type=bool)
    parser.add_argument("--streaming", default=False, type=bool)
    parser.add_argument("--num_training_data", default=388892357, type=int)
    args = parser.parse_args()

    # val_check_interval
    if "." in args.val_check_interval:
        args.val_check_interval = float(args.val_check_interval)
    else:
        args.val_check_interval = int(args.val_check_interval)

    if args.max_epochs is None or args.max_epochs < 0:
        args.max_epochs = 1000000
    if args.max_steps is None and args.max_epochs is None:
        raise ValueError("max_steps and max_epochs can not be None at the same time")

    if args.warmup_max_steps is None or (args.warmup_max_steps < args.max_steps):
        print("warmup_max_steps should be larger than max_steps")
        print("Set warmup_max_steps to max_steps")
        args.warmup_max_steps = args.max_steps

    return args


def get_trainer(args):
    callbacks = get_finetuning_callbacks(
        save_model_dir=args.save_model_dir,
        save_model_name=args.save_model_name,
        log_gradient_norm=args.log_gradient_norm,
        log_parameter=False,
        save_interval=args.save_interval,
        patience=args.patience,
    )

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

        print(f"accumulate_grad_batches is set to: {args.accumulate_grad_batches}")

    else:
        args.global_train_batch_size = (
            args.train_batch_size
            * args.nodes
            * args.devices
            * args.accumulate_grad_batches
        )
    monitor_patience_callback = MonitorPatienceCallback()
    trainer = Trainer(
        strategy=strategy,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks + [monitor_patience_callback],
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
    )

    return trainer


def main():
    args = parse_args()
    if args.pretrain_task == "seq2seq":
        tokenizer = PromptTokenizer()
    trainer = get_trainer(args)
    if args.seed is not None:
        seed_everything(args.seed)
    datamodule = FinetuningDataModule(
        tokenizer=tokenizer,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        collator=args.pretrain_task,
        noise_p=0.15,
        to_mask_p=0.8,
        to_random_p=0.1,
        streaming=args.streaming,
        num_workers=args.num_workers,
        args=args,
    )
    model = LightningProLM(
        config_path=args.config_path,
        pretrain_task=args.pretrain_task,
        lr=args.lr,
        adam_epsilon=args.adam_epsilon,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_steps=args.warmup_steps,
        warmup_max_steps=args.warmup_max_steps,
        weight_dacay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        eval_determinstic=False,
        encoder_ckpt=args.encoder_ckpt,
        args=args,
        tokenizer=tokenizer,
        cds_gen=args.cds_gen,
        save_output_result=args.save_output_result,
    )

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
