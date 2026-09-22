import os, json
import torch
from utils.data_utils import tokenize, numericalize
from transformer_model import TransformerNL2SQL

BASE = os.path.dirname(os.path.abspath(__file__))


def load_transformer(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = TransformerNL2SQL(
        src_vocab_size=len(ckpt["src_vocab"]),
        trg_vocab_size=len(ckpt["trg_vocab"]),
        d_model=ckpt["d_model"],
        n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"],
        d_ff=ckpt["d_ff"],
        dropout=ckpt["dropout"],
        max_len=ckpt.get("max_len", 512),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt["src_vocab"], ckpt["trg_vocab"], ckpt["trg_idx2word"]


def predict_sql(model, question, src_vocab, trg_vocab,
                trg_idx2word, device, max_len=60):
    src_ids = numericalize(tokenize(question), src_vocab)
    src     = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

    sos_idx  = trg_vocab.get("<sos>", 1)
    eos_idx  = trg_vocab.get("<eos>", 2)
    pad_idx  = trg_vocab.get("<pad>", 0)
    semi_idx = trg_vocab.get(";", -1)
    sel_idx  = trg_vocab.get("SELECT", None)

    src_mask = model.make_src_mask(src)
    with torch.no_grad():
        enc_out = model.encode(src, src_mask)

    generated = [sos_idx]

    if sel_idx is not None:
        generated.append(sel_idx)

    prev   = None
    consec = 0

    for _ in range(max_len):
        trg_tensor = torch.tensor(
            generated, dtype=torch.long
        ).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            dec_out = model.decode(trg_tensor, enc_out, trg_mask, src_mask)
            logits  = model.fc_out(dec_out[:, -1, :])

        logits[0, pad_idx] = float("-inf")
        logits[0, sos_idx] = float("-inf")

        top1  = logits.argmax(1).item()
        token = trg_idx2word.get(top1, "<unk>")

        if top1 == eos_idx:
            break
        if top1 == semi_idx:
            generated.append(top1)
            break
        if top1 == prev:
            consec += 1
            if consec >= 5:
                break
        else:
            consec = 0

        generated.append(top1)
        prev = top1

    tokens = [trg_idx2word.get(idx, "<unk>")
              for idx in generated[1:]]

    return " ".join(tokens)


def evaluate_exact_match(model, samples, src_vocab, trg_vocab,
                          trg_idx2word, device, verbose=True):
    correct = 0
    results = []

    for item in samples:
        pred = predict_sql(
            model, item["question"],
            src_vocab, trg_vocab, trg_idx2word, device
        )
        gold = item["sql"].strip()
        ok   = pred.strip().lower() == gold.lower()
        if ok:
            correct += 1

        results.append({
            "question": item["question"],
            "gold": gold, "predicted": pred, "correct": ok,
        })

        if verbose:
            print(f"\n{'✅' if ok else '❌'} Q: {item['question']}")
            if not ok:
                print(f"   Gold: {gold}")
                print(f"   Pred: {pred}")

    acc = correct / len(samples) if samples else 0.0
    return acc, results


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    ckpt_path = os.path.join(
        BASE, "outputs", "checkpoints", "best_transformer.pt"
    )
    model, src_vocab, trg_vocab, trg_idx2word = load_transformer(
        ckpt_path, device
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint: epoca {ckpt.get('epoch')}, "
          f"valid_loss={ckpt.get('valid_loss')}")
    print(f"Arhitectura: d_model={ckpt['d_model']}, "
          f"n_heads={ckpt['n_heads']}, n_layers={ckpt['n_layers']}\n")

    demo = [
        "câte consultații au fost în 2023 ?",
        "arată toți pacienții",
        "câți medici sunt ?",
        "medici din cluj-napoca",
        "consultațiile pacientului ionescu",
        "care este costul total al consultațiilor ?",
        "pacienți cu vârsta peste 50 ani",
        "arată pacienții bărbați",
        "medici de cardiologie",
        "cel mai experimentat medic",
    ]

    print("=" * 60)
    print("  Demo predicții - Transformer")
    print("=" * 60)
    for q in demo:
        sql = predict_sql(
            model, q, src_vocab, trg_vocab, trg_idx2word, device
        )
        print(f"\nQ: {q}")
        print(f"S: {sql}")

    test_path = os.path.join(BASE, "..", "data", "test.json")
    if os.path.exists(test_path):
        print("\n\n" + "=" * 60)
        print("  Evaluare Exact Match - Transformer")
        print("=" * 60)
        with open(test_path, encoding="utf-8") as f:
            test_samples = json.load(f)

        acc, results = evaluate_exact_match(
            model, test_samples, src_vocab, trg_vocab,
            trg_idx2word, device, verbose=True
        )
        n_ok = sum(r["correct"] for r in results)
        print(f"\n{'-'*60}")
        print(f"  Exact Match: {acc:.1%} ({n_ok}/{len(results)})")

        out_path = os.path.join(
            BASE, "..", "outputs", "transformer_results.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "Transformer",
                "accuracy": acc, "n_correct": n_ok,
                "n_total": len(results), "results": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"  Salvat: {out_path}")