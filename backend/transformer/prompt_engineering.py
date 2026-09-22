import os, json, re, time

os.environ["GEMINI_API_KEY"] = ""

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "outputs")

SCHEMA = """
Schema bazei de date medicale:

pacienti(
  id INTEGER PRIMARY KEY,
  nume TEXT, prenume TEXT,
  varsta INTEGER,
  sex TEXT CHECK(sex IN ('M', 'F')),
  oras TEXT, telefon TEXT
)

medici(
  id INTEGER PRIMARY KEY,
  nume TEXT, prenume TEXT,
  specialitate TEXT,
  experienta_ani INTEGER,
  oras TEXT
)

consultatii(
  id INTEGER PRIMARY KEY,
  pacient_id INTEGER REFERENCES pacienti(id),
  medic_id INTEGER REFERENCES medici(id),
  data TEXT,
  diagnostic TEXT,
  cost REAL,
  durata_minute INTEGER
)
"""

FEW_SHOT_EXAMPLES = [
    {
        "question": "câți pacienți sunt ?",
        "sql": "SELECT COUNT ( * ) AS total FROM pacienti ;",
    },
    {
        "question": "medici din cluj-napoca",
        "sql": "SELECT * FROM medici WHERE oras = 'Cluj-Napoca' ;",
    },
    {
        "question": "pacienți cu vârsta peste 50 ani",
        "sql": "SELECT * FROM pacienti WHERE varsta > 50 ;",
    },
    {
        "question": "consultațiile pacientului ionescu",
        "sql": (
            "SELECT p.nume , p.prenume , c.data , c.diagnostic , c.cost "
            "FROM pacienti p JOIN consultatii c ON c.pacient_id = p.id "
            "WHERE p.nume = 'Ionescu' OR p.prenume = 'Ionescu' ;"
        ),
    },
    {
        "question": "care este costul total al consultațiilor ?",
        "sql": "SELECT SUM ( cost ) AS total_cost FROM consultatii ;",
    },
    {
        "question": "medici de cardiologie",
        "sql": "SELECT * FROM medici WHERE specialitate = 'Cardiologie' ;",
    },
]

def build_zero_shot_prompt(question: str) -> str:
    return f"""Ești un expert SQL. Generează DOAR interogarea SQL pentru întrebarea dată, fără explicații.

{SCHEMA}

Reguli:
- Folosește exact numele tabelelor și coloanelor din schemă
- Termină interogarea cu ;
- Pune spații în jurul operatorilor: COUNT ( * ), varsta > 50

Întrebare: {question}
SQL:"""


def build_few_shot_prompt(question: str, n_examples: int = 3) -> str:
    examples_text = ""
    for ex in FEW_SHOT_EXAMPLES[:n_examples]:
        examples_text += f"\nÎntrebare: {ex['question']}\n"
        examples_text += f"SQL: {ex['sql']}\n"

    return f"""Ești un expert SQL. Generează DOAR interogarea SQL, fără explicații.

{SCHEMA}

Exemple:
{examples_text}
Întrebare: {question}
SQL:"""


def build_cot_prompt(question: str) -> str:
    return f"""Ești un expert SQL. Analizează întrebarea pas cu pas, apoi generează SQL-ul.

{SCHEMA}

Exemplu de raționament:
Întrebare: "câți medici de cardiologie din cluj există ?"
Pas 1 - Tabele: avem nevoie de tabela 'medici'
Pas 2 - Operație: numărare → COUNT(*)
Pas 3 - Condiții: specialitate = 'Cardiologie' AND oras = 'Cluj-Napoca'
Pas 4 - SQL: SELECT COUNT(*) AS total FROM medici WHERE specialitate = 'Cardiologie' AND oras = 'Cluj-Napoca' ;

Acum analizează:
Întrebare: "{question}"
Pas 1 - Tabele:
Pas 2 - Operație:
Pas 3 - Condiții:
Pas 4 - SQL:"""

def call_api(prompt: str, model_id: str = "gemini-3.1-flash-lite") -> str:
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "EROARE: Cheia API lipsește."

        client = genai.Client(api_key=api_key)
        time.sleep(4)

        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=300,
                system_instruction="Ești un expert SQL. Returnează DOAR codul SQL, fără formatare Markdown sau text adițional."
            ),
        )

        if response and response.text:
            text = response.text.strip()
            if text.startswith("```"):
                text = text.replace("```sql", "").replace("```", "").strip()
            return text

        return ""

    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            print(f"Modelul '{model_id}' nu a fost găsit. Încearcă 'gemini-3.1-flash-lite'.")
        elif "429" in error_msg:
            print("Limită de rată atinsă. Reîncearcă peste 60 de secunde.")
        else:
            print(f"Eroare neașteptată: {e}")
        return ""


def extract_sql(response: str) -> str:
    code_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", response,
                           re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()

    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("SELECT"):
            return line

    sql_match = re.search(r"Pas 4.*?SQL:\s*(.+?)(?:\n|$)",
                          response, re.IGNORECASE | re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()

    return response.strip()

def evaluate_technique(technique: str, samples: list,
                       n_examples: int = 3,
                       verbose: bool = True) -> dict:
    correct = 0
    results = []

    for item in samples:
        q = item["question"]
        gold = item["sql"].strip()

        if technique == "zero_shot":
            prompt = build_zero_shot_prompt(q)
        elif technique == "few_shot":
            prompt = build_few_shot_prompt(q, n_examples)
        elif technique == "cot":
            prompt = build_cot_prompt(q)
        else:
            raise ValueError(f"Tehnică necunoscută: {technique}")

        response = call_api(prompt)
        pred = extract_sql(response)

        ok = pred.strip().lower() == gold.lower()
        if ok:
            correct += 1

        results.append({
            "question": q, "gold": gold,
            "predicted": pred, "correct": ok,
            "raw_response": response,
        })

        if verbose:
            print(f"\n{'✅' if ok else '❌'} Q: {q}")
            if not ok:
                print(f"   Gold: {gold}")
                print(f"   Pred: {pred}")

        time.sleep(0.5)

    acc = correct / len(samples) if samples else 0.0
    return {
        "technique": technique,
        "accuracy": acc,
        "n_correct": correct,
        "n_total": len(samples),
        "results": results,
    }


def run_demo():
    demo_questions = [
        "câți medici sunt ?",
        "arată toți pacienții",
        "medici din cluj-napoca",
        "pacienți cu vârsta peste 50 ani",
        "care este costul total al consultațiilor ?",
    ]

    print("=" * 65)
    print("  Demo Prompt Engineering — NL2SQL")
    print("=" * 65)

    for technique in ["zero_shot", "few_shot", "cot"]:
        labels = {
            "zero_shot": "Zero-shot",
            "few_shot": "Few-shot (3 exemple)",
            "cot": "Chain-of-Thought",
        }
        print(f"\n{'─' * 65}")
        print(f"  Tehnică: {labels[technique]}")
        print(f"{'─' * 65}")

        for q in demo_questions[:3]:
            if technique == "zero_shot":
                prompt = build_zero_shot_prompt(q)
            elif technique == "few_shot":
                prompt = build_few_shot_prompt(q, 3)
            else:
                prompt = build_cot_prompt(q)

            response = call_api(prompt)
            sql = extract_sql(response)

            print(f"\nQ: {q}")
            print(f"S: {sql}")
            time.sleep(0.3)


def run_full_evaluation():
    test_path = os.path.join(DATA_DIR, "test.json")
    if not os.path.exists(test_path):
        print(f"test.json negăsit: {test_path}")
        return

    with open(test_path, encoding="utf-8") as f:
        test_samples = json.load(f)

    test_subset = test_samples[:20]
    print(f"Evaluare pe {len(test_subset)} exemple din test set\n")

    all_results = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for technique in ["zero_shot", "few_shot", "cot"]:
        labels = {
            "zero_shot": "Zero-shot",
            "few_shot": "Few-shot",
            "cot": "Chain-of-Thought",
        }
        print(f"\n{'=' * 65}")
        print(f"  Evaluare: {labels[technique]}")
        print(f"{'=' * 65}")

        metrics = evaluate_technique(
            technique, test_subset, verbose=True
        )
        all_results[technique] = metrics

        print(f"\n  Exact Match ({labels[technique]}): "
              f"{metrics['accuracy']:.1%} "
              f"({metrics['n_correct']}/{metrics['n_total']})")

    out_path = os.path.join(OUTPUT_DIR, "prompt_engineering_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n\n" + "=" * 65)
    print("  COMPARAȚIE TEHNICI PROMPT ENGINEERING")
    print("=" * 65)
    print(f"  {'Tehnică':<25} {'Exact Match':>12}  {'Bar'}")
    print("─" * 65)
    for tech, metrics in all_results.items():
        acc = metrics["accuracy"]
        bar = "█" * int(acc * 30) + "░" * (30 - int(acc * 30))
        print(f"  {labels[tech]:<25} {acc:>11.1%}  {bar}")
    print("=" * 65)
    print(f"\nRezultate salvate: {out_path}")


if __name__ == "__main__":
    import sys

    if "--full" in sys.argv:
        run_full_evaluation()
    else:
        run_demo()