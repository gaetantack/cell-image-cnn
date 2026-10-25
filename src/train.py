"""Train one CNN and score it on the validation split. One entry point for all CNN runs.

Example:
    python -m src.train --model smallcnn --classes 4 --seed 0 --lr 5e-4 --run-tag final
    python -m src.train --model smallcnn --smoke-test

What one run does:
1. Load brightfield + darkfield images (src.data) and the fixed train/val rows (src.splits). Never test.
2. Optionally keep a stratified fraction of the TRAIN rows (--label-fraction, learning curve).
3. Normalise each channel with the mean and SD of the (kept) training images only.
4. Train with AdamW and a cosine learning-rate schedule; after every epoch score val macro-F1,
   keep the best weights, and stop early if val macro-F1 has not improved for EARLY_STOP_PATIENCE epochs.
5. Score the best weights on val through src.evaluate, append one row to results/experiments.csv,
   save val predictions to results/preds/<run_id>.npz and the best weights to checkpoints/<run_id>_best.pt.

A checkpoint is written after every epoch, so an interrupted run continues with --resume.
A run whose run_id is already in the experiment log is skipped.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src import config
from src.data import load_dataset
from src.evaluate import evaluate
from src.experiments import has_run, log_experiment
from src.models import MODEL_NAMES, build_model, count_parameters
from src.splits import load_split

CLASS_NAMES = {4: config.CLASSES_4, 7: config.CLASSES_7}


def get_device():
    """cuda if available, else Apple-Silicon mps, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the mps/cuda generators


def make_run_id(args):
    return f"{args.model}_c{args.classes}_{args.run_tag}_s{args.seed}"


def subsample_train(train, y, fraction, seed):
    """Stratified fraction of the training rows (at least one cell per class)."""
    if fraction >= 1.0:
        return train
    keep, _ = train_test_split(train, train_size=fraction, stratify=y[train], random_state=seed)
    for c in np.unique(y[train]):  # guarantee every class is present
        if not (y[keep] == c).any():
            keep = np.append(keep, train[y[train] == c][0])
    return np.sort(keep)


def channel_stats(X_uint8):
    """Per-channel mean and SD over all pixels of the given (training) images."""
    x = torch.from_numpy(X_uint8).float()
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))


def to_input(xb_uint8, mean, sd):
    """uint8 batch on the device -> normalised float batch."""
    return (xb_uint8.float() - mean.view(1, -1, 1, 1)) / sd.view(1, -1, 1, 1)


def augment_basic(x, gen):
    """Random 90-degree rotation, horizontal and vertical flips, small intensity jitter, per image.

    Random numbers come from the seeded CPU generator `gen`, so a run is reproducible.
    Rotations and flips are exact (no interpolation), and cells have no natural orientation.
    """
    n = x.shape[0]
    k = torch.randint(0, 4, (n,), generator=gen)
    flip_h = torch.rand(n, generator=gen) < 0.5
    flip_v = torch.rand(n, generator=gen) < 0.5
    gain = (torch.rand(n, x.shape[1], 1, 1, generator=gen) * 2 - 1) * config.AUG_JITTER
    bias = (torch.rand(n, x.shape[1], 1, 1, generator=gen) * 2 - 1) * config.AUG_JITTER
    out = torch.empty_like(x)
    for r in range(4):
        idx = (k == r).nonzero().flatten().to(x.device)
        if len(idx):
            out[idx] = torch.rot90(x[idx], r, dims=(2, 3))
    fh, fv = flip_h.to(x.device).view(-1, 1, 1, 1), flip_v.to(x.device).view(-1, 1, 1, 1)
    out = torch.where(fh, out.flip(3), out)
    out = torch.where(fv, out.flip(2), out)
    return out * (1 + gain.to(x.device)) + bias.to(x.device)


def class_weights(y_train, n_classes):
    """Inverse-frequency weights N / (K * count_k): every class contributes equally to the loss."""
    counts = np.bincount(y_train, minlength=n_classes)
    return torch.tensor(len(y_train) / (n_classes * counts), dtype=torch.float32)


def epoch_order(y_train, balance, gen, n_draws):
    """Order of training cells for one epoch (n_draws cells). 'oversample' draws cells with probability
    1/class count (with replacement), so every class is seen about equally often; otherwise random
    permutations of the training cells, repeated if n_draws is larger than the number of cells."""
    n = len(y_train)
    if balance == "oversample":
        p = 1.0 / np.bincount(y_train)[y_train]
        return torch.multinomial(torch.tensor(p / p.sum()), n_draws, replacement=True, generator=gen)
    n_repeats = int(np.ceil(n_draws / n))
    return torch.cat([torch.randperm(n, generator=gen) for _ in range(n_repeats)])[:n_draws]


@torch.no_grad()
def predict(model, X_dev, mean, sd):
    """Predicted class for every image (X_dev: uint8 tensor on the device)."""
    model.eval()
    preds = []
    for i in range(0, len(X_dev), config.EVAL_BATCH_SIZE):
        preds.append(model(to_input(X_dev[i:i + config.EVAL_BATCH_SIZE], mean, sd)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_NAMES, required=True)
    p.add_argument("--classes", type=int, choices=(4, 7), default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=config.FINAL_EPOCHS)
    p.add_argument("--augment", choices=("none", "basic"), default="basic")
    p.add_argument("--balance", choices=("none", "weighted", "oversample"), default="weighted")
    p.add_argument("--label-fraction", type=float, default=1.0)
    p.add_argument("--run-tag", default="manual", help="part of the run_id, e.g. tune03, final, lf0.05")
    p.add_argument("--smoke-test", action="store_true", help="tiny subset, 1 epoch, nothing logged")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<run_id>_last.pt")
    p.add_argument("--device", default=None, help="override automatic device choice")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke_test:
        args.epochs, args.run_tag = 1, "smoke"
    run_id = make_run_id(args)
    if not args.smoke_test and has_run(run_id):
        print(f"skip {run_id}: already in {config.EXPERIMENTS_FILE.name}")
        return None
    device = torch.device(args.device) if args.device else get_device()
    seed_everything(args.seed)
    gen = torch.Generator().manual_seed(args.seed)  # all training randomness (order, augmentation)

    # Data: train and val rows only
    X, y, _ = load_dataset(args.classes)
    train, val = load_split("train"), load_split("val")
    # One "epoch" always shows as many images as the full training split has cells, so a run on a
    # small label fraction gets the same number of gradient steps as a full run (its cells repeat).
    samples_per_epoch = len(train)
    train = subsample_train(train, y, args.label_fraction, args.seed)
    if args.smoke_test:
        rng = np.random.default_rng(0)
        train, val = np.sort(rng.choice(train, 256, replace=False)), np.sort(rng.choice(val, 128, replace=False))
        samples_per_epoch = len(train)
    mean, sd = channel_stats(X[train])
    mean, sd = mean.to(device), sd.to(device)
    X_train = torch.from_numpy(X[train]).to(device)  # kept as uint8 on the device to save memory
    X_val = torch.from_numpy(X[val]).to(device)
    y_train_np, y_val = y[train], y[val]
    y_train = torch.from_numpy(y_train_np).to(device)

    model = build_model(args.model, args.classes).to(device)
    weight = class_weights(y_train_np, args.classes).to(device) if args.balance == "weighted" else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(samples_per_epoch / args.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    ckpt_dir = config.CHECKPOINT_DIR if not args.smoke_test else config.CHECKPOINT_DIR / "smoke"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / f"{run_id}_last.pt", ckpt_dir / f"{run_id}_best.pt"
    start_epoch, best_f1, best_epoch, elapsed, history = 0, -1.0, -1, 0.0, []
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"])
        start_epoch, best_f1, best_epoch = ck["epoch"] + 1, ck["best_f1"], ck["best_epoch"]
        elapsed, history = ck["elapsed"], ck["history"]
        print(f"resumed {run_id} at epoch {start_epoch}")

    print(f"{run_id}: {args.model}, {count_parameters(model):,} parameters, device {device}, "
          f"{len(train)} train / {len(val)} val cells")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        model.train()
        order = epoch_order(y_train_np, args.balance, gen, samples_per_epoch)
        total_loss = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size].to(device)
            xb = to_input(X_train[idx], mean, sd)
            if args.augment == "basic":
                xb = augment_basic(xb, gen)
            loss = loss_fn(model(xb), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(idx)
        val_f1 = evaluate(y_val, predict(model, X_val, mean, sd), CLASS_NAMES[args.classes])["macro_f1"]
        elapsed += time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": total_loss / len(order), "val_macro_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), best_path)
        print(f"  epoch {epoch}: train loss {total_loss / len(order):.4f}  val macro-F1 {val_f1:.4f}  "
              f"({elapsed:.0f} s)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch,
                    "elapsed": elapsed, "history": history}, last_path)
        if epoch - best_epoch >= config.EARLY_STOP_PATIENCE:
            print(f"  early stop: no improvement since epoch {best_epoch}")
            break

    # Final score: best weights on val
    model.load_state_dict(torch.load(best_path, map_location=device))
    pred = predict(model, X_val, mean, sd)
    m = evaluate(y_val, pred, CLASS_NAMES[args.classes])
    print(f"{run_id}: best epoch {best_epoch}, val macro-F1 {m['macro_f1']:.4f}, "
          f"bal. acc {m['balanced_accuracy']:.4f}, train time {elapsed:.0f} s")
    if args.smoke_test:
        return m
    settings = {
        "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "max_epochs": args.epochs, "epochs_run": len(history), "best_epoch": best_epoch,
        "augment": args.augment, "balance": args.balance, "label_fraction": args.label_fraction,
        "n_train": len(train), "samples_per_epoch": samples_per_epoch,
        "n_params": count_parameters(model), "run_tag": args.run_tag,
        "input_size": getattr(model, "input_size", config.IMAGE_SIZE),
    }
    log_experiment(args.model, args.classes, "val", args.seed, settings, m, elapsed,
                   run_id=run_id, device=str(device))
    config.PREDS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.PREDS_DIR / f"{run_id}.npz", y_true=y_val, y_pred=pred, rows=val)
    (config.LOG_DIR / "history").mkdir(parents=True, exist_ok=True)
    (config.LOG_DIR / "history" / f"{run_id}.json").write_text(json.dumps(history))
    last_path.unlink(missing_ok=True)  # the run is finished; keep only the best weights
    return m


if __name__ == "__main__":
    main()
