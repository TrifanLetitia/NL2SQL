import os
import re
import json
import time
import argparse


os.environ["GEMINI_API_KEY"] = ""

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)

DB_CONFIG = {
    "server":             "localhost",
    "database":           "SpitalDB",
    "driver":             "ODBC Driver 17 for SQL Server",
    "trusted_connection": "yes",
    "username": "",
    "password": "",
}

def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_sql(sql: str) -> str:
    if not sql:
        return ""
    sql = sql.lower().strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*\(\s*", " ( ", sql)
    sql = re.sub(r"\s*\)\s*", " ) ", sql)
    sql = re.sub(r"\s*,\s*", " , ", sql)
    sql = re.sub(r"\s*;\s*$", "", sql)
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip()


def get_db_connection():
    try:
        import pyodbc
        if DB_CONFIG["trusted_connection"].lower() == "yes":
            conn_str = (
                f"DRIVER={{{DB_CONFIG['driver']}}};"
                f"SERVER={DB_CONFIG['server']};"
                f"DATABASE={DB_CONFIG['database']};"
                f"Trusted_Connection=yes;"
            )
        else:
            conn_str = (
                f"DRIVER={{{DB_CONFIG['driver']}}};"
                f"SERVER={DB_CONFIG['server']};"
                f"DATABASE={DB_CONFIG['database']};"
                f"UID={DB_CONFIG['username']};"
                f"PWD={DB_CONFIG['password']};"
            )
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"  [WARN] Conexiune BD eșuată: {e}")
        print("  EA și VES nu vor fi calculate.\n")
        return None


def timed_execute(conn, sql: str):
    if not sql or conn is None:
        return None, "no_connection_or_sql", 0.0
    try:
        cursor = conn.cursor()
        t0 = time.perf_counter()
        cursor.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.perf_counter() - t0
        return [tuple(r) for r in rows], None, elapsed
    except Exception as e:
        return None, str(e), 0.0


def rows_equal(pred_rows, gold_rows) -> bool:
    if pred_rows is None or gold_rows is None:
        return False
    try:
        return sorted(pred_rows) == sorted(gold_rows)
    except TypeError:
        return pred_rows == gold_rows


def compute_ves(gold_time: float, pred_time: float,
                is_correct: bool) -> float:
    if not is_correct:
        return 0.0
    if pred_time <= 0:
        return 1.0
    return min((gold_time / pred_time) ** 0.5, 1.0)

def predict_template(question: str, translator) -> str:
    try:
        result = translator.translate(question)
        return result.get("sql") or ""
    except Exception:
        return ""


def predict_grammar(question: str, grammar_sys) -> str:
    try:
        result = grammar_sys.translate(question)
        return result.get("sql") or ""
    except Exception:
        return ""


def predict_seq2seq(question: str, model, src_vocab, trg_vocab,
                    trg_idx2word, device, max_len: int = 50) -> str:
    import torch
    from utils.data_utils import tokenize, numericalize

    try:
        src_pad_idx = src_vocab["<pad>"]
        src_ids = numericalize(tokenize(question), src_vocab)

        if len(src_ids) < max_len:
            src_ids += [src_pad_idx] * (max_len - len(src_ids))
        else:
            src_ids = src_ids[:max_len]

        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            enc_out, hidden, cell = model.encoder(src)

        inp = torch.tensor([trg_vocab["<sos>"]], dtype=torch.long).to(device)
        tokens = []

        for _ in range(max_len):
            with torch.no_grad():
                out, hidden, cell, _ = model.decoder(inp, hidden, cell, enc_out)
            top1  = out.argmax(1).item()
            token = trg_idx2word.get(top1, "<unk>")
            if token in ("<eos>",):
                break
            if token not in ("<pad>", "<sos>", "<eos>"):
                tokens.append(token)
            inp = torch.tensor([top1], dtype=torch.long).to(device)
            if token == ";":
                break

        return " ".join(tokens)
    except Exception:
        return ""


def predict_transformer(question: str, model, src_vocab, trg_vocab,
                        trg_idx2word, device, max_len: int = 60) -> str:
    import torch
    from utils.data_utils import tokenize, numericalize

    try:
        src_ids = numericalize(tokenize(question), src_vocab)
        src     = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

        sos_idx  = trg_vocab.get("<sos>", 1)
        eos_idx  = trg_vocab.get("<eos>", 2)
        pad_idx  = trg_vocab.get("<pad>", 0)
        sel_idx  = trg_vocab.get("SELECT")

        src_mask = model.make_src_mask(src)
        with torch.no_grad():
            enc_out = model.encode(src, src_mask)

        generated = [sos_idx]
        if sel_idx is not None:
            generated.append(sel_idx)

        for _ in range(max_len):
            trg_t    = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_t)
            with torch.no_grad():
                dec_out = model.decode(trg_t, enc_out, trg_mask, src_mask)
                logits  = model.fc_out(dec_out[:, -1, :])
            logits[0, pad_idx] = float("-inf")
            logits[0, sos_idx] = float("-inf")
            top1 = logits.argmax(1).item()
            if top1 == eos_idx:
                break
            generated.append(top1)
            if trg_idx2word.get(top1) == ";":
                break

        tokens = [trg_idx2word.get(i, "<unk>") for i in generated[1:]]
        return " ".join(tokens)
    except Exception:
        return ""


def predict_mt5(question: str, model, tokenizer, device,
                max_len: int = 128) -> str:
    try:
        input_text = f"translate to SQL: {question}"
        inputs = tokenizer(
            input_text, return_tensors="pt",
            max_length=max_len, truncation=True
        ).to(device)
        with __import__("torch").no_grad():
            outputs = model.generate(
                **inputs, max_length=max_len,
                num_beams=4, early_stopping=True,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception:
        return ""


def predict_prompt(question: str, technique: str = "cot") -> str:
    try:
        import re as _re
        from google import genai
        from google.genai import types

        SCHEMA = """
pacienti(id, nume, prenume, varsta, sex, oras, telefon)
medici(id, nume, prenume, specialitate, experienta_ani, oras)
consultatii(id, pacient_id, medic_id, data, diagnostic, cost, durata_minute)
"""
        FEW_SHOT = [
            ("câți pacienți sunt ?",
             "SELECT COUNT ( * ) AS total FROM pacienti ;"),
            ("medici din cluj-napoca",
             "SELECT * FROM medici WHERE oras = 'Cluj-Napoca' ;"),
            ("pacienți cu vârsta peste 50 ani",
             "SELECT * FROM pacienti WHERE varsta > 50 ;"),
            ("consultațiile pacientului ionescu",
             "SELECT p.nume , p.prenume , c.data , c.diagnostic , c.cost "
             "FROM pacienti p JOIN consultatii c ON c.pacient_id = p.id "
             "WHERE p.nume = 'Ionescu' OR p.prenume = 'Ionescu' ;"),
            ("care este costul total al consultațiilor ?",
             "SELECT SUM ( cost ) AS total_cost FROM consultatii ;"),
            ("medici de cardiologie",
             "SELECT * FROM medici WHERE specialitate = 'Cardiologie' ;"),
        ]

        if technique == "zero_shot":
            prompt = (
                f"Ești expert SQL. Generează DOAR SQL, fără explicații.\n"
                f"Schema: {SCHEMA}\n"
                f"Reguli: termină cu ; și folosește spații în jurul operatorilor.\n"
                f"Întrebare: {question}\nSQL:"
            )
        elif technique == "few_shot":
            ex = "\n".join(
                f"Întrebare: {q}\nSQL: {s}" for q, s in FEW_SHOT
            )
            prompt = (
                f"Ești expert SQL. Generează DOAR SQL, fără explicații.\n"
                f"Schema: {SCHEMA}\nExemple:\n{ex}\n"
                f"Întrebare: {question}\nSQL:"
            )
        else:  # cot
            prompt = (
                f"Ești expert SQL. Analizează pas cu pas, apoi generează SQL.\n"
                f"Schema: {SCHEMA}\n"
                f"Exemplu:\nÎntrebare: câți medici de cardiologie din cluj ?\n"
                f"Pas 1 - Tabele: medici\nPas 2 - Operație: COUNT(*)\n"
                f"Pas 3 - Condiții: specialitate='Cardiologie' AND oras='Cluj-Napoca'\n"
                f"Pas 4 - SQL: SELECT COUNT(*) AS total FROM medici "
                f"WHERE specialitate='Cardiologie' AND oras='Cluj-Napoca' ;\n\n"
                f"Întrebare: {question}\nPas 1 - Tabele:\nPas 2 - Operație:\n"
                f"Pas 3 - Condiții:\nPas 4 - SQL:"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("EROARE API: GEMINI_API_KEY nu este setată.")
            return ""

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=256,
            ),
        )

        text = (response.text or "").strip()

        code_m = _re.search(r"```(?:sql)?\s*(.*?)\s*```", text,
                            _re.DOTALL | _re.IGNORECASE)
        if code_m:
            return code_m.group(1).strip()
        for line in text.split("\n"):
            if line.strip().upper().startswith("SELECT"):
                return line.strip()
        cot_m = _re.search(r"Pas 4.*?SQL:\s*(.+?)(?:\n|$)",
                           text, _re.IGNORECASE | _re.DOTALL)
        if cot_m:
            return cot_m.group(1).strip()
        return text

    except Exception as e:
        print(f"[PROMPT ERROR] {e}")
        return ""

def evaluate_method(method_name: str, predict_fn, samples: list,
                    conn=None, verbose: bool = False) -> dict:
    em_ok = ea_ok = 0
    ves_scores = []
    results = []
    use_db = conn is not None

    for item in samples:
        question = item["question"]
        gold_sql = item["sql"].strip()

        pred_sql = predict_fn(question)

        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        if em:
            em_ok += 1

        ea  = False
        ves = 0.0
        gold_time = pred_time = 0.0
        gold_err  = pred_err  = None

        if use_db:
            gold_rows, gold_err, gold_time = timed_execute(conn, gold_sql)
            pred_rows, pred_err, pred_time = timed_execute(conn, pred_sql)
            ea  = rows_equal(pred_rows, gold_rows)
            ves = compute_ves(gold_time, pred_time, ea)
            if ea:
                ea_ok += 1

        ves_scores.append(ves)

        results.append({
            "question":           question,
            "gold":               gold_sql,
            "predicted":          pred_sql,
            "exact_match":        em,
            "execution_accuracy": ea,
            "ves":                round(ves, 4),
            "gold_time":          round(gold_time, 6),
            "pred_time":          round(pred_time, 6),
            "gold_error":         gold_err,
            "pred_error":         pred_err,
        })

        if verbose:
            icon = "✅" if em else "❌"
            print(f"{icon} Q: {question}")
            if not em:
                print(f"   Gold: {gold_sql}")
                print(f"   Pred: {pred_sql}")
            if use_db and pred_err:
                print(f"   Eroare exec: {pred_err}")

    n = len(samples)
    return {
        "method":              method_name,
        "n_total":             n,
        "exact_match":         round(em_ok / n, 4) if n else 0.0,
        "n_exact":             em_ok,
        "execution_accuracy":  round(ea_ok / n, 4) if (n and use_db) else None,
        "n_exec":              ea_ok if use_db else None,
        "ves":                 round(sum(ves_scores) / n, 4) if (n and use_db) else None,
        "results":             results,
    }

def load_seq2seq(ckpt_path: str, device):
    import torch
    from seq2seq.src.model import Encoder, Attention, Decoder, Seq2Seq

    ckpt = torch.load(ckpt_path, map_location=device)
    enc  = Encoder(len(ckpt["src_vocab"]), ckpt["emb_dim"], ckpt["hidden_dim"])
    att  = Attention(ckpt["hidden_dim"])
    dec  = Decoder(len(ckpt["trg_vocab"]), ckpt["emb_dim"], ckpt["hidden_dim"], att)
    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["src_vocab"], ckpt["trg_vocab"], ckpt["trg_idx2word"]


def load_transformer(ckpt_path: str, device):
    import torch
    from transformer.transformer_model import TransformerNL2SQL

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = TransformerNL2SQL(
        src_vocab_size=len(ckpt["src_vocab"]),
        trg_vocab_size=len(ckpt["trg_vocab"]),
        d_model=ckpt["d_model"],   n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"], d_ff=ckpt["d_ff"],
        dropout=ckpt["dropout"],   max_len=ckpt.get("max_len", 60),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["src_vocab"], ckpt["trg_vocab"], ckpt["trg_idx2word"]


def load_mt5(model_dir: str, device):
    from transformers import ( MT5ForConditionalGeneration, AutoTokenizer )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = MT5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer

def evaluate_prompt_manual(
    json_path: str,
    conn=None,
    out_path: str = None,
) -> dict:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    techniques = ["zero_shot", "few_shot", "cot"]
    use_db = conn is not None

    all_results = {}

    for tech in techniques:
        em_ok = ea_ok = 0
        ves_scores = []
        results = []

        for item in data:
            gold_sql  = item["sql_gold"].strip()
            tech_data = item.get(tech, {})
            pred_sql  = (tech_data.get("sql_generat") or "").strip()

            em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
            if em:
                em_ok += 1

            ea = False
            ves = 0.0
            gold_time = pred_time = 0.0
            gold_err = pred_err = None

            if use_db:
                gold_rows, gold_err, gold_time = timed_execute(conn, gold_sql)
                pred_rows, pred_err, pred_time = timed_execute(conn, pred_sql)
                ea  = rows_equal(pred_rows, gold_rows)
                ves = compute_ves(gold_time, pred_time, ea)
                if ea:
                    ea_ok += 1

            ves_scores.append(ves)

            results.append({
                "nr":                 item["nr"],
                "question":           item["question"],
                "gold":               gold_sql,
                "predicted":          pred_sql,
                "exact_match":        em,
                "execution_accuracy": ea if use_db else None,
                "ves":                round(ves, 4) if use_db else None,
                "gold_time":          round(gold_time, 6),
                "pred_time":          round(pred_time, 6),
                "gold_error":         gold_err,
                "pred_error":         pred_err,
            })

        n = len(data)
        all_results[tech] = {
            "technique":           tech,
            "n_total":             n,
            "exact_match":         round(em_ok / n, 4) if n else 0.0,
            "n_exact":             em_ok,
            "execution_accuracy":  round(ea_ok / n, 4) if (n and use_db) else None,
            "n_exec":              ea_ok if use_db else None,
            "ves":                 round(sum(ves_scores) / n, 4) if (n and use_db) else None,
            "results":             results,
        }

    labels = {
        "zero_shot": "Zero-shot",
        "few_shot":  "Few-shot",
        "cot":       "Chain-of-Thought",
    }


def evaluate_prompt_manual(json_path, conn=None, out_path=None):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    techniques = ["zero_shot", "few_shot", "cot"]
    use_db = conn is not None
    all_results = {}

    for tech in techniques:
        em_ok = ea_ok = 0
        ves_scores = []
        results = []

        for item in data:
            gold_sql = item["sql_gold"].strip()
            tech_data = item.get(tech, {})
            pred_sql = (tech_data.get("sql_generat") or "").strip()

            em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
            if em:
                em_ok += 1

            ea = False
            ves = 0.0
            gold_time = pred_time = 0.0
            gold_err = pred_err = None

            if use_db:
                gold_rows, gold_err, gold_time = timed_execute(conn, gold_sql)
                pred_rows, pred_err, pred_time = timed_execute(conn, pred_sql)
                ea = rows_equal(pred_rows, gold_rows)
                ves = compute_ves(gold_time, pred_time, ea)
                if ea:
                    ea_ok += 1

            ves_scores.append(ves)

            results.append({
                "nr":                 item["nr"],
                "question":           item["question"],
                "gold":               gold_sql,
                "predicted":          pred_sql,
                "exact_match":        em,
                "execution_accuracy": ea if use_db else None,
                "ves":                round(ves, 4) if use_db else None,
                "gold_time":          round(gold_time, 6),
                "pred_time":          round(pred_time, 6),
                "gold_error":         gold_err,
                "pred_error":         pred_err,
            })

        n = len(data)
        all_results[tech] = {
            "technique":           tech,
            "n_total":             n,
            "exact_match":         round(em_ok / n, 4) if n else 0.0,
            "n_exact":             em_ok,
            "execution_accuracy":  round(ea_ok / n, 4) if (n and use_db) else None,
            "n_exec":              ea_ok if use_db else None,
            "ves":                 round(sum(ves_scores) / n, 4) if (n and use_db) else None,
            "results":             results,
        }

    labels = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "cot": "Chain-of-Thought"}

    print("=" * 65)
    print("  REZULTATE PROMPT ENGINEERING -- 20 exemple")
    print("=" * 65)
    hdr = "  {:<20} {:>8}".format("Tehnica", "EM (%)")
    if use_db:
        hdr += " {:>8} {:>8}".format("EA (%)", "VES")
    print(hdr)
    print("-" * 65)

    for tech, m in all_results.items():
        em = "{:.1f}".format(m["exact_match"] * 100)
        line = "  {:<20} {:>8}".format(labels[tech], em)
        if use_db:
            ea = "{:.1f}".format(m["execution_accuracy"] * 100) if m["execution_accuracy"] is not None else "N/A"
            ves = "{:.4f}".format(m["ves"]) if m["ves"] is not None else "N/A"
            line += " {:>8} {:>8}".format(ea, ves)
        print(line)

    print("=" * 65)

    if out_path is None:
        out_path = os.path.join(OUT_DIR, "prompt_manual_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\nSalvat: {}".format(out_path))

    return all_results


def main_prompt_manual(json_path):
    conn = get_db_connection()
    evaluate_prompt_manual(json_path, conn=conn)
    if conn:
        conn.close()

def main():
    parser = argparse.ArgumentParser(
        description="Evaluare NL2SQL - toate cele 6 metode"
    )
    parser.add_argument(
        "--method", default="all",
        choices=["all", "template", "grammar",
                 "seq2seq", "transformer", "mt5",
                 "prompt_zero", "prompt_few", "prompt_cot"],
        help="Metoda de evaluat (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Numărul maxim de exemple din test set (default: toate)"
    )
    parser.add_argument(
        "--db", default=None,
        help="Calea către baza de date"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Afișează detalii pentru fiecare exemplu"
    )
    args = parser.parse_args()

    test_path = os.path.join(DATA_DIR, "test.json")
    samples   = load_json(test_path)
    if args.limit:
        samples = samples[:args.limit]
    print(f"Evaluare pe {len(samples)} exemple din test set\n")

    conn = None if getattr(args, "no_db", False) else get_db_connection()
    if conn:
        print("Conexiune BD realizată - EA și VES vor fi calculate.")
    else:
        print("BD negăsită - se calculează doar Exact Match.\n")

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = None
    print(f"Device: {device}\n")

    all_metrics = {}
    methods_to_run = (
        ["template", "grammar", "seq2seq",
         "transformer", "mt5",
         "prompt_zero", "prompt_few", "prompt_cot"]
        if args.method == "all" else [args.method]
    )

    for method in methods_to_run:
        print(f"{'='*60}")
        print(f"  Metodă: {method.upper()}")
        print(f"{'='*60}")

        predict_fn = None

        if method == "template":
            try:
                from templates.translator import NL2SQLTranslator
                translator = NL2SQLTranslator()
                predict_fn = lambda q: predict_template(q, translator)
            except ImportError:
                print("  [SKIP] utils.translator nu a fost găsit.")
                continue

        elif method == "grammar":
            try:
                from grammar.grammar_nl2sql import GrammarNL2SQL
                grammar_sys = GrammarNL2SQL(db_conn=conn)
                predict_fn  = lambda q: predict_grammar(q, grammar_sys)
            except ImportError:
                print("  [SKIP] grammar_nl2sql nu a fost găsit.")
                continue

        elif method == "seq2seq":
            try:
                ckpt_path = os.path.join(
                    OUT_DIR, "checkpoints", "best_seq2seq_attention.pt"
                )
                model, sv, tv, ti = load_seq2seq(ckpt_path, device)
                predict_fn = lambda q: predict_seq2seq(
                    q, model, sv, tv, ti, device
                )
            except Exception as e:
                print(f"  [SKIP] Seq2Seq: {e}")
                continue

        elif method == "transformer":
            try:
                ckpt_path = os.path.join(
                    OUT_DIR, "checkpoints", "best_transformer.pt"
                )
                model, sv, tv, ti = load_transformer(ckpt_path, device)
                predict_fn = lambda q: predict_transformer(
                    q, model, sv, tv, ti, device
                )
            except Exception as e:
                print(f"  [SKIP] Transformer: {e}")
                continue

        elif method == "mt5":
            try:
                model_dir  = os.path.join(
                    OUT_DIR, "checkpoints", "mt5_finetuned"
                )
                model, tok = load_mt5(model_dir, device)
                predict_fn = lambda q: predict_mt5(q, model, tok, device)
            except Exception as e:
                print(f"  [SKIP] mT5: {e}")
                continue

        elif method in ("prompt_zero", "prompt_few", "prompt_cot"):
            technique  = method.replace("prompt_", "")
            predict_fn = lambda q, t=technique: predict_prompt(q, t)

        if predict_fn is None:
            continue

        metrics = evaluate_method(
            method_name=method,
            predict_fn=predict_fn,
            samples=samples,
            conn=conn,
            verbose=args.verbose,
        )
        all_metrics[method] = metrics

        em  = metrics["exact_match"]
        ea  = metrics["execution_accuracy"]
        ves = metrics["ves"]
        n   = metrics["n_total"]
        print(f"\n  Exact Match:        {em:.1%} ({metrics['n_exact']}/{n})")
        if ea is not None:
            print(f"  Execution Accuracy: {ea:.1%} ({metrics['n_exec']}/{n})")
            print(f"  VES adaptat:        {ves:.4f}")
        print()

    out_json = os.path.join(OUT_DIR, "evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    labels = {
        "template":    "Template-based",
        "grammar":     "Grammar PCFG",
        "seq2seq":     "Seq2Seq LSTM",
        "transformer": "Transformer",
        "mt5":         "mT5 fine-tuned",
        "prompt_zero": "Prompt Zero-shot",
        "prompt_few":  "Prompt Few-shot",
        "prompt_cot":  "Prompt CoT",
    }

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("  REZULTATE COMPARATIVE - NL2SQL")
    summary_lines.append("=" * 70)
    summary_lines.append(
        f"  {'Metodă':<22} {'EM (%)':>8} {'EA (%)':>8} {'VES':>8}"
    )
    summary_lines.append("─" * 70)

    for method, m in all_metrics.items():
        em  = f"{m['exact_match']*100:.1f}"
        ea  = f"{m['execution_accuracy']*100:.1f}" if m["execution_accuracy"] is not None else "N/A"
        ves = f"{m['ves']:.4f}" if m["ves"] is not None else "N/A"
        summary_lines.append(
            f"  {labels.get(method, method):<22} {em:>8} {ea:>8} {ves:>8}"
        )

    summary_lines.append("=" * 70)
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    out_txt = os.path.join(OUT_DIR, "evaluation_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"\nRezultate complete: {out_json}")
    print(f"Tabel comparativ:   {out_txt}")

    if conn:
        conn.close()



if __name__ == "__main__":
    import sys
    if "--prompt-manual" in sys.argv:
        idx = sys.argv.index("--prompt-manual")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "evaluation_prompt.json"
        main_prompt_manual(path)
    else:
        main()
