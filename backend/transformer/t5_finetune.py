import os, json, time
import torch
from torch.utils.data import Dataset, DataLoader

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "data")
OUTPUT_DIR = os.path.join(BASE, "outputs")
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
MODEL_DIR  = os.path.join(CKPT_DIR,   "mt5_finetuned")

MODEL_NAME  = "google/mt5-small"
MAX_SRC_LEN = 128
MAX_TRG_LEN = 128
BATCH_SIZE  = 8
N_EPOCHS    = 3
LR          = 3e-4
PATIENCE    = 4

class MT5SQLDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        question = item["question"]
        sql = item["sql"]

        input_text = f"translate to SQL: {question}"

        source = self.tokenizer(
            input_text,
            max_length=MAX_SRC_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target = self.tokenizer(
            text_target=sql,
            max_length=MAX_TRG_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels
        }

def finetune():
    try:
        from transformers import (MT5ForConditionalGeneration,
                                   T5Tokenizer, AutoTokenizer)
    except ImportError:
        print("EROARE: Instalează transformers:")
        print("  pip install transformers sentencepiece")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("  Fine-tuning mT5-small pe date medicale (română)")
    print("=" * 60)

    def load_json(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    train_samples = load_json(os.path.join(DATA_DIR, "train.json"))
    valid_samples = load_json(os.path.join(DATA_DIR, "valid.json"))
    print(f"Train: {len(train_samples)} | Valid: {len(valid_samples)}")

    print(f"\nÎncărcare {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model     = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri: {n_params:,}")

    train_loader = DataLoader(
        MT5SQLDataset(train_samples, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    valid_loader = DataLoader(
        MT5SQLDataset(valid_samples, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_loss  = float("inf")
    no_improve = 0
    history    = {"train_loss": [], "valid_loss": []}

    print(f"\n{'Epoca':>6}  {'Train':>9}  {'Valid':>9}  {'Timp':>6}")
    print("-" * 40)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += outputs.loss.item()
        valid_loss = val_loss / max(len(valid_loader), 1)
        elapsed    = time.time() - t0

        scheduler.step(valid_loss)
        history["train_loss"].append(round(train_loss, 4))
        history["valid_loss"].append(round(valid_loss, 4))

        marker = ""
        if valid_loss < best_loss:
            best_loss  = valid_loss
            no_improve = 0
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            marker = "  ✓"
        else:
            no_improve += 1

        print(f"{epoch:>6}  {train_loss:>9.4f}  {valid_loss:>9.4f}  "
              f"{elapsed:>5.1f}s{marker}")

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping la epoca {epoch}.")
            break

    with open(os.path.join(OUTPUT_DIR, "mt5_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Best valid loss: {best_loss:.4f}")
    print(f"Model salvat: {MODEL_DIR}")


def evaluate():
    try:
        from transformers import (MT5ForConditionalGeneration,
                                   AutoTokenizer)
    except ImportError:
        print("pip install transformers sentencepiece")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(MODEL_DIR):
        print(f"Model negăsit: {MODEL_DIR}")
        return

    print(f"Încărcare model din {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model     = model.to(device)
    model.eval()

    def predict(question: str) -> str:
        input_text = f"translate to SQL: {question}"
        inputs = tokenizer(
            input_text, return_tensors="pt",
            max_length=128, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    demo = [
        "câți medici sunt ?",
        "arată toți pacienții",
        "medici din cluj-napoca",
        "pacienți cu vârsta peste 50 ani",
        "care este costul total al consultațiilor ?",
    ]

    print("\n" + "=" * 60)
    print("  Demo predicții - mT5-small fine-tuned")
    print("=" * 60)
    for q in demo:
        sql = predict(q)
        print(f"\nQ: {q}")
        print(f"S: {sql}")

    test_path = os.path.join(DATA_DIR, "test.json")
    if os.path.exists(test_path):
        with open(test_path, encoding="utf-8") as f:
            test_samples = json.load(f)

        correct = 0
        results = []
        print("\n" + "=" * 60)
        print("  Evaluare Exact Match - mT5-small")
        print("=" * 60)

        for item in test_samples:
            pred = predict(item["question"])
            gold = item["sql"].strip()
            ok   = pred.strip().lower() == gold.lower()
            if ok:
                correct += 1
            results.append({
                "question": item["question"],
                "gold": gold, "predicted": pred, "correct": ok,
            })
            print(f"\n{'✅' if ok else '❌'} Q: {item['question']}")
            if not ok:
                print(f"   Gold: {gold}")
                print(f"   Pred: {pred}")

        acc  = correct / len(test_samples)
        n_ok = correct
        print(f"\n{'-'*60}")
        print(f"  Exact Match mT5: {acc:.1%} ({n_ok}/{len(test_samples)})")

        out_path = os.path.join(OUTPUT_DIR, "mt5_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "mT5-small fine-tuned",
                "accuracy": acc, "n_correct": n_ok,
                "n_total": len(results), "results": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"  Salvat: {out_path}")


if __name__ == "__main__":
    import sys
    if "--eval" in sys.argv:
        evaluate()
    else:
        finetune()