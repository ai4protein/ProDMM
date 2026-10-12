"""Train or evaluate ProDMM seq2seq using paired source/target JSONL data."""

import argparse
from pathlib import Path


def strtobool(value):
    """Parse explicit booleans; unlike bool('false'), this returns False."""
    normalized = value.lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test"), default="test")
    parser.add_argument(
        "--config_path", required=True, help="Config directory or model ID"
    )
    parser.add_argument(
        "--pretrain_ckpt", help="Optional pretrained seq2seq checkpoint"
    )
    parser.add_argument(
        "--encoder_ckpt", help="Optional encoder initialization checkpoint"
    )
    parser.add_argument(
        "--trainer_ckpt", help="Lightning checkpoint to resume or evaluate"
    )
    parser.add_argument("--train_file", help="Source/target JSONL training data")
    parser.add_argument("--valid_file", help="Source/target JSONL validation data")
    parser.add_argument("--test_file", help="Source/target JSONL evaluation data")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--streaming", type=strtobool, nargs="?", const=True, default=False
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--scheduler_type", default="linear")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_value", type=float, default=1.0)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--save_model_dir", default="checkpoint")
    parser.add_argument("--save_model_name", default="prodmm_seq2seq")
    parser.add_argument(
        "--patience", type=int, help="Validation epochs before early stopping"
    )
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument(
        "--logger", choices=("none", "tensorboard", "wandb"), default="none"
    )
    parser.add_argument("--logger_project", default="prodmm")
    parser.add_argument("--logger_run_name", default="seq2seq")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--pretrain_task", choices=("seq2seq",), default="seq2seq")
    parser.add_argument(
        "--task_type",
        choices=("prot_cds", "cds_promoter", "promoter_cds"),
        default="prot_cds",
    )
    for flag in ("frozen_encoder", "cds_gen"):
        parser.add_argument(
            f"--{flag}", type=strtobool, nargs="?", const=True, default=False
        )
    args = parser.parse_args(argv)
    if args.mode == "train" and (not args.train_file or not args.valid_file):
        parser.error("--mode train requires --train_file and --valid_file")
    if args.mode == "test" and not args.test_file:
        parser.error("--mode test requires --test_file")
    if args.streaming and (args.devices > 1 or args.nodes > 1):
        parser.error(
            "Streaming requires a single process; distributed node sharding is not implemented"
        )
    if args.streaming and args.mode == "train" and args.max_steps <= 0:
        parser.error("Streaming training requires a positive --max_steps")
    return args


def get_logger(args, output_dir):
    if args.logger == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger

        return TensorBoardLogger(output_dir, name=args.logger_run_name)
    if args.logger == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(
            name=args.logger_run_name,
            project=args.logger_project,
            entity=args.wandb_entity,
        )
    return False


def main(argv=None):
    args = parse_args(argv)
    # Keep CLI help available without installing the optional training extra.
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from prodmm.seq2seq.datamodule import FinetuningDataModule
    from prodmm.seq2seq.lightning_module_finetuning import Seq2SeqModule
    from prodmm.seq2seq_tokenizer.prompttokenizer import PromptTokenizer

    seed_everything(args.seed, workers=True)
    output_dir = Path(args.save_model_dir) / args.save_model_name
    data = FinetuningDataModule(
        PromptTokenizer(),
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        streaming=args.streaming,
    )
    model = Seq2SeqModule(
        config_path=args.config_path,
        checkpoint=args.pretrain_ckpt,
        encoder_checkpoint=args.encoder_ckpt,
        frozen_encoder=args.frozen_encoder,
        cds_gen=args.cds_gen,
        task_type=args.task_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        scheduler_type=args.scheduler_type,
        warmup_steps=args.warmup_steps,
    )
    callbacks = []
    if args.mode == "train":
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename="best",
                monitor="val/loss_epoch",
                mode="min",
                save_last=True,
            )
        )
        if args.patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss_epoch", patience=args.patience, mode="min"
                )
            )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        strategy=args.strategy,
        precision=args.precision,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_value,
        log_every_n_steps=args.log_steps,
        logger=get_logger(args, output_dir),
        callbacks=callbacks,
        enable_checkpointing=args.mode == "train",
        default_root_dir=output_dir,
    )
    if args.mode == "train":
        trainer.fit(model, datamodule=data, ckpt_path=args.trainer_ckpt)
        # Save after the fit loop finishes so resume starts from its final state.
        trainer.save_checkpoint(output_dir / "last.ckpt")
        if trainer.is_global_zero:
            model.model.save_pretrained(output_dir / "model")
    else:
        trainer.test(model, datamodule=data, ckpt_path=args.trainer_ckpt)


if __name__ == "__main__":
    main()
