from functools import partial
import os, json, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_utils import load_json, build_vocab, TextToSQLDataset, collate_fn
from transformer_model import TransformerNL2SQL

BASE         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE, "data")
OUTPUT_DIR   = os.path.join(BASE, "outputs")
CKPT_DIR     = os.path.join(OUTPUT_DIR, "checkpoints")
CKPT_PATH    = os.path.join(CKPT_DIR, "best_transformer.pt")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "transformer_history.json")

D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 3
D_FF     = 512
DROPOUT  = 0.1
MAX_LEN  = 60

BATCH_SIZE = 32
N_EPOCHS   = 100
WARMUP     = 400
PATIENCE   = 15


class WarmupScheduler:

    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.step_num     = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup_steps ** -1.5
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def evaluate_loss(model, loader, criterion, device, pad_idx):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for src, trg in loader:
            src, trg   = src.to(device), trg.to(device)
            trg_in  = trg[:, :-1]
            trg_out = trg[:, 1:]

            logits = model(src, trg_in)
            V      = logits.shape[-1]
            loss   = criterion(
                logits.reshape(-1, V),
                trg_out.reshape(-1)
            )
            total += loss.item()
    return total / max(len(loader), 1)

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print("=" * 60)
    print("  Antrenare Transformer NL2SQL")
    print("=" * 60)

    train_samples = load_json(os.path.join(DATA_DIR, "train.json"))
    valid_samples = load_json(os.path.join(DATA_DIR, "valid.json"))
    print(f"Train: {len(train_samples)} | Valid: {len(valid_samples)}")

    src_vocab, src_idx2word = build_vocab(train_samples, "question")
    trg_vocab, trg_idx2word = build_vocab(train_samples, "sql")
    pad_idx = trg_vocab["<pad>"]
    print(f"Vocab src: {len(src_vocab)} | Vocab trg: {len(trg_vocab)}")

    collate = partial(
        collate_fn,
        src_pad_idx=src_vocab["<pad>"],
        trg_pad_idx=pad_idx,
        max_len=MAX_LEN,
    )
    train_loader = DataLoader(
        TextToSQLDataset(train_samples, src_vocab, trg_vocab),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
    )
    valid_loader = DataLoader(
        TextToSQLDataset(valid_samples, src_vocab, trg_vocab),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
    )

    model = TransformerNL2SQL(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF,
        dropout=DROPOUT, max_len=MAX_LEN,
    ).to(device)

    print(f"Parametri: {model.n_params():,}")
    print(f"Arhitectura: d_model={D_MODEL}, n_heads={N_HEADS}, "
          f"n_layers={N_LAYERS}, d_ff={D_FF}\n")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler  = WarmupScheduler(optimizer, D_MODEL, WARMUP)
    criterion  = nn.CrossEntropyLoss(
        ignore_index=pad_idx, label_smoothing=0.1
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_loss  = float("inf")
    no_improve = 0
    history    = {"train_loss": [], "valid_loss": [], "lr": []}

    print(f"{'Epoca':>6}  {'Train':>9}  {'Valid':>9}  "
          f"{'LR':>10}  {'Timp':>6}")
    print("-" * 55)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)

            trg_in  = trg[:, :-1]
            trg_out = trg[:, 1:]

            optimizer.zero_grad()
            logits = model(src, trg_in)
            V      = logits.shape[-1]
            loss   = criterion(logits.reshape(-1, V), trg_out.reshape(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            current_lr = scheduler.step()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        valid_loss = evaluate_loss(
            model, valid_loader, criterion, device, pad_idx
        )
        elapsed    = time.time() - t0

        history["train_loss"].append(round(train_loss, 4))
        history["valid_loss"].append(round(valid_loss, 4))
        history["lr"].append(round(current_lr, 6))

        marker = ""
        if valid_loss < best_loss:
            best_loss  = valid_loss
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "src_vocab":        src_vocab,
                "trg_vocab":        trg_vocab,
                "src_idx2word":     src_idx2word,
                "trg_idx2word":     trg_idx2word,
                "d_model":          D_MODEL,
                "n_heads":          N_HEADS,
                "n_layers":         N_LAYERS,
                "d_ff":             D_FF,
                "dropout":          DROPOUT,
                "max_len":          MAX_LEN,
                "epoch":            epoch,
                "valid_loss":       round(valid_loss, 4),
            }, CKPT_PATH)
            marker = "  ✓"
        else:
            no_improve += 1

        print(f"{epoch:>6}  {train_loss:>9.4f}  {valid_loss:>9.4f}  "
              f"{current_lr:>10.6f}  {elapsed:>5.1f}s{marker}")

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping la epoca {epoch}.")
            break

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest valid loss: {best_loss:.4f}")
    print(f"Checkpoint: {CKPT_PATH}")


if __name__ == "__main__":
    main()