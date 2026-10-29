"""Command line for preparation, training, evaluation, and prediction."""

from __future__ import annotations

import argparse
import sys

from covid_cxr.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_LR,
    DEFAULT_LR_FINETUNE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DISCLAIMER,
)
from covid_cxr.discovery import DatasetNotFoundError


def _add_data_args(parser: argparse.ArgumentParser, include_seed: bool) -> None:
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Folder that contains the COVID and Normal directories. Searched automatically when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where manifests, checkpoints, metrics, and plots are written. Default: outputs/",
    )
    if include_seed:
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed. Default: 42.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m covid_cxr",
        description="Educational COVID vs Normal chest X-ray classifier. " + DISCLAIMER,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Validate images and write a reusable split manifest.")
    _add_data_args(prepare, include_seed=True)
    prepare.add_argument(
        "--resplit",
        action="store_true",
        help="Replace an existing manifest. Without this flag, a matching manifest is reused.",
    )

    train = subparsers.add_parser("train", help="Smoke-test the pipeline or run full training.")
    _add_data_args(train, include_seed=True)
    mode = train.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true", help="Train the head for one epoch on a few images.")
    mode.add_argument("--full", action="store_true", help="Train on the full training split.")
    train.add_argument("--head-only", action="store_true", help="Skip fine-tuning the final residual block.")
    train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate while training the new head.")
    train.add_argument(
        "--lr-finetune",
        type=float,
        default=DEFAULT_LR_FINETUNE,
        help="Smaller learning rate after the final residual block is unfrozen.",
    )
    train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum epochs for the head.")
    train.add_argument(
        "--finetune-epochs",
        type=int,
        default=DEFAULT_FINETUNE_EPOCHS,
        help="Maximum epochs after unfreezing the final residual block.",
    )
    train.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early-stopping patience per phase.")
    train.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)

    evaluate = subparsers.add_parser("evaluate", help="Freeze a validation threshold, then score the test set.")
    _add_data_args(evaluate, include_seed=False)
    evaluate.add_argument("--checkpoint", default=None, help="Defaults to outputs/checkpoints/best.pt")
    evaluate.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    evaluate.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)

    predict = subparsers.add_parser("predict", help="Score one X-ray with the saved checkpoint.")
    predict.add_argument("--image", required=True, help="Path to one chest X-ray image.")
    predict.add_argument("--checkpoint", default=None)
    predict.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from covid_cxr.prepare import prepare

            prepare(data_dir=args.data_dir, output_dir=args.output_dir, seed=args.seed, resplit=args.resplit)
        elif args.command == "train":
            if not args.smoke_test and not args.full:
                print("This project does not start full training unless you ask for it.")
                print("Smoke test, recommended first:")
                print("  python -m covid_cxr train --smoke-test")
                print("Full training:")
                print("  python -m covid_cxr train --full")
                return 2
            from covid_cxr.train import train

            train(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                seed=args.seed,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_finetune=args.lr_finetune,
                epochs=args.epochs,
                finetune_epochs=args.finetune_epochs,
                patience=args.patience,
                num_workers=args.num_workers,
                head_only=args.head_only,
                smoke_test=args.smoke_test,
                full=args.full,
            )
        elif args.command == "evaluate":
            from covid_cxr.evaluate import evaluate

            evaluate(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                checkpoint=args.checkpoint,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        elif args.command == "predict":
            from covid_cxr.predict import predict

            predict(image=args.image, checkpoint=args.checkpoint, output_dir=args.output_dir)
        else:
            parser.error(f"Unknown command: {args.command}")
    except (DatasetNotFoundError, RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
