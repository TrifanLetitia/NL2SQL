import os, sys, time, re
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

'''
BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE, "src")
sys.path.insert(0, SRC)
'''

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

BASE = resource_path("")
SRC = resource_path("src")
sys.path.insert(0, SRC)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

try:
    from templates.translator import NL2SQLTranslator
    template_model = NL2SQLTranslator()
    print("[OK] Template-based")
except Exception as e:
    print(f"[ERROR] Template-based: {e}")
    template_model = None

try:
    from grammar.grammar_nl2sql import GrammarNL2SQL
    grammar_model = GrammarNL2SQL()
    print("[OK] Grammar-based")
except Exception as e:
    print(f"[ERROR] Grammar-based: {e}")
    grammar_model = None

try:
    from transformers import MT5ForConditionalGeneration, AutoTokenizer
    mt5_dir       = os.path.join(BASE, "outputs", "checkpoints", "mt5_finetuned")
    mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_dir)
    mt5_model     = MT5ForConditionalGeneration.from_pretrained(mt5_dir).to(device)
    mt5_model.eval()
    print("[OK] mT5 fine-tuned")
except Exception as e:
    print(f"[ERROR] mT5: {e}")
    mt5_model     = None
    mt5_tokenizer = None

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "AIzaSyDXr43n01wzVvDaq0uaQ4JG6u73ptxU4o8")
gemini_available = bool(GEMINI_API_KEY)
if gemini_available:
    print("[OK] Prompt Engineering (Gemini API)")
else:
    print("[ERROR] Prompt Engineering: variabila GEMINI_API_KEY lipsește")



def load_seq2seq(ckpt_path: str, device):
    import torch
    from seq2seq.src.model import Encoder, Attention, Decoder, Seq2Seq

    ckpt  = torch.load(ckpt_path, map_location=device)
    enc   = Encoder(len(ckpt["src_vocab"]), ckpt["emb_dim"], ckpt["hidden_dim"])
    att   = Attention(ckpt["hidden_dim"])
    dec   = Decoder(len(ckpt["trg_vocab"]), ckpt["emb_dim"], ckpt["hidden_dim"], att)
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
        d_model =ckpt["d_model"],   n_heads =ckpt["n_heads"],
        n_layers=ckpt["n_layers"],  d_ff    =ckpt["d_ff"],
        dropout =ckpt["dropout"],   max_len =ckpt.get("max_len", 60),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["src_vocab"], ckpt["trg_vocab"], ckpt["trg_idx2word"]

seq2seq_model = None
seq2seq_sv    = None
seq2seq_tv    = None
seq2seq_ti    = None

transformer_model = None
transformer_sv    = None
transformer_tv    = None
transformer_ti    = None

try:
    seq2seq_model, seq2seq_sv, seq2seq_tv, seq2seq_ti = load_seq2seq(
        os.path.join(BASE, "outputs", "checkpoints", "best_seq2seq_attention.pt"),
        device
    )
    print("[OK] Seq2Seq LSTM")
except Exception as e:
    print(f"[ERROR] Seq2Seq: {e}")

try:
    transformer_model, transformer_sv, transformer_tv, transformer_ti = load_transformer(
        os.path.join(BASE, "outputs", "checkpoints", "best_transformer.pt"),
        device
    )
    print("[OK] Transformer")
except Exception as e:
    print(f"[ERROR] Transformer: {e}")


def predict_seq2seq(question: str, model, src_vocab, trg_vocab,
                    trg_idx2word, device, max_len: int = 50) -> str:
    import torch
    from utils.data_utils import tokenize, numericalize

    try:
        src_pad_idx = src_vocab["<pad>"]
        src_ids     = numericalize(tokenize(question), src_vocab)

        if len(src_ids) < max_len:
            src_ids += [src_pad_idx] * (max_len - len(src_ids))
        else:
            src_ids = src_ids[:max_len]

        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            enc_out, hidden, cell = model.encoder(src)

        inp    = torch.tensor([trg_vocab["<sos>"]], dtype=torch.long).to(device)
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
        src_ids  = numericalize(tokenize(question), src_vocab)
        src      = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

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


def predict_mt5(question: str) -> str:
    if not mt5_model:
        raise RuntimeError("Model mT5 indisponibil")
    try:
        inputs = mt5_tokenizer(
            f"translate to SQL: {question}",
            return_tensors="pt", max_length=128, truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = mt5_model.generate(
                **inputs, max_length=128, num_beams=4, early_stopping=True
            )
        return mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception:
        return ""

SCHEMA_TEXT = """
Table pacienti (
  id INTEGER PRIMARY KEY,
  nume TEXT, prenume TEXT, varsta INTEGER,
  sex TEXT CHECK (sex IN ('M', 'F')),
  oras TEXT, telefon TEXT
)
Table medici (
  id INTEGER PRIMARY KEY,
  nume TEXT, prenume TEXT, specialitate TEXT,
  experienta_ani INTEGER, oras TEXT
)
Table consultatii (
  id INTEGER PRIMARY KEY,
  pacient_id INTEGER REFERENCES pacienti(id),
  medic_id INTEGER REFERENCES medici(id),
  data TEXT, diagnostic TEXT, cost REAL, durata_minute INTEGER
)
"""

FEW_SHOT_EXAMPLES = [
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
]

def build_zero_shot_prompt(question: str) -> str:
    return (
        f"Ești un expert SQL. Generează DOAR interogarea SQL, fără explicații.\n\n"
        f"Schema:{SCHEMA_TEXT}\n"
        f"Reguli:\n"
        f"- Folosește exact numele tabelelor și coloanelor din schemă\n"
        f"- Pune spații în jurul operatorilor: COUNT ( * ), varsta > 50\n"
        f"- Folosește sintaxă SQL Server validă\n"
        f"- Nu inventa tabele sau coloane care nu apar în schemă\n"
        f"- Termină cu ;\n\n"
        f"Întrebare: {question}\nSQL:"
    )

def build_few_shot_prompt(question: str) -> str:
    ex = "\n".join(f"Întrebare: {q}\nSQL: {s}" for q, s in FEW_SHOT_EXAMPLES)
    return (
        f"Ești un expert SQL. Generează DOAR SQL-ul, fără explicații.\n\n"
        f"Schema:{SCHEMA_TEXT}\n"
        f"Reguli:\n"
        f"- Folosește sintaxă SQL Server validă\n"
        f"- Nu inventa tabele sau coloane care nu apar în schemă\n"
        f"- Termină cu ;\n\n"
        f"Exemple:\n{ex}\n\n"
        f"Întrebare: {question}\nSQL:"
    )

def build_cot_prompt(question: str) -> str:
    return (
        f"Ești un expert SQL. Analizează intern pașii necesari, "
        f"dar returnează doar SQL-ul final.\n\n"
        f"Schema:{SCHEMA_TEXT}\n"
        f"Reguli:\n"
        f"- Folosește sintaxă SQL Server validă\n"
        f"- Nu inventa tabele sau coloane care nu apar în schemă\n"
        f"- Termină cu ;\n\n"
        f"Exemplu raționament:\n"
        f"Întrebare: \"câți medici de cardiologie din cluj există ?\"\n"
        f"Pas 1 - Tabele: medici\n"
        f"Pas 2 - Operație: numărare → COUNT(*)\n"
        f"Pas 3 - Condiții: specialitate='Cardiologie' AND oras='Cluj-Napoca'\n"
        f"Pas 4 - SQL: SELECT COUNT ( * ) AS total FROM medici "
        f"WHERE specialitate = 'Cardiologie' AND oras = 'Cluj-Napoca' ;\n\n"
        f"Întrebare: \"{question}\"\n"
        f"Pas 1 - Tabele:\nPas 2 - Operație:\nPas 3 - Condiții:\nPas 4 - SQL:"
    )

def extract_sql(response: str) -> str:
    if not response:
        return ""
    text = response.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    text = re.sub(r"^(SQL|Răspuns|Query)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    m = re.search(r"\b(SELECT|WITH)\b.*?;", text, re.DOTALL | re.IGNORECASE)
    if m:
        return " ".join(m.group(0).split())
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith(("SELECT", "WITH")):
            return line if line.endswith(";") else line + " ;"
    return text if text.endswith(";") else text + " ;"

def call_gemini(prompt: str) -> str:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        time.sleep(4)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=300,
                system_instruction=(
                    "Ești un expert SQL. Returnează DOAR codul SQL, "
                    "fără formatare Markdown sau text adițional."
                )
            ),
        )
        return extract_sql(response.text.strip()) if response and response.text else ""
    except Exception as e:
        print(f"Eroare Gemini API: {e}")
        return ""

def predict_prompt(question: str, technique: str = "few_shot") -> str:
    if technique == "zero_shot":
        prompt = build_zero_shot_prompt(question)
    elif technique == "cot":
        prompt = build_cot_prompt(question)
    else:
        prompt = build_few_shot_prompt(question)
    return call_gemini(prompt)

@app.route("/predict", methods=["POST"])
def predict():
    data      = request.get_json()
    question  = question = (data.get("question") or "").strip().lower()
    model_id  = data.get("model", "transformer")
    technique = data.get("technique", "few_shot")

    if not question:
        return jsonify({"error": "Întrebarea lipsește"}), 400

    t0 = time.perf_counter()
    try:
        if model_id == "template":
            if not template_model:
                raise RuntimeError("Model template indisponibil")
            result = template_model.translate(question)
            if isinstance(result, dict):
                sql = result.get("sql") or result.get("error", "Niciun șablon potrivit")
            else:
                sql = str(result)

        elif model_id == "grammar":
            if not grammar_model:
                raise RuntimeError("Model grammar indisponibil")
            result = grammar_model.translate(question)
            if isinstance(result, dict):
                sql = result.get("sql") or result.get("error", "Niciun rezultat")
            else:
                sql = str(result)

        elif model_id == "seq2seq":
            if seq2seq_model is None:
                raise RuntimeError("Model Seq2Seq indisponibil")
            sql = predict_seq2seq(
                question,
                seq2seq_model, seq2seq_sv, seq2seq_tv, seq2seq_ti,
                device
            )

        elif model_id == "transformer":
            if transformer_model is None:
                raise RuntimeError("Model Transformer indisponibil")
            sql = predict_transformer(
                question,
                transformer_model, transformer_sv, transformer_tv, transformer_ti,
                device
            )

        elif model_id == "mt5":
            sql = predict_mt5(question)

        elif model_id == "prompt":
            if not gemini_available:
                raise RuntimeError("Cheia GEMINI_API_KEY lipsește")
            sql = predict_prompt(question, technique)

        else:
            return jsonify({"error": f"Model necunoscut: {model_id}"}), 400

    except Exception as e:
        return jsonify({"error": str(e), "sql": None}), 200

    return jsonify({
        "question":  question,
        "sql":       sql,
        "model":     model_id,
        "technique": technique if model_id == "prompt" else None,
        "time_ms":   round((time.perf_counter() - t0) * 1000, 1),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "template":    template_model    is not None,
            "grammar":     grammar_model     is not None,
            "seq2seq":     seq2seq_model     is not None,
            "transformer": transformer_model is not None,
            "mt5":         mt5_model         is not None,
            "prompt":      gemini_available,
        }
    })


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Server NL2SQL pornit la http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)