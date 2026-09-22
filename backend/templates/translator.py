from utils.schema import get_connection, build_schema_info
from templates.engine import TEMPLATES
from templates.stratica_templates import match_stratica_templates
from templates.nlp import normalize

class NL2SQLTranslator:
    def __init__(self, db_conn=None):
        if db_conn is None:
            self.conn = get_connection()
        else:
            self.conn = db_conn

        self.schema_info = build_schema_info(self.conn)

    def translate(self, question: str) -> dict:
        norm = normalize(question)

        stratica = match_stratica_templates(question, norm)
        if stratica and stratica.get("sql"):
            return {
                "question":  question,
                "sql":       stratica["sql"],
                "template":  stratica["tip"],
                "tip":       stratica["tip"],
                "descriere": stratica["descriere"],
                "slots":     stratica.get("slots", {}),
                "error":     None,
            }

        for tmpl in TEMPLATES:
            match = tmpl.match(norm)
            if match:
                result = tmpl.generate(question, norm, match)
                if result:
                    return {
                        "question":  question,
                        "sql":       result["sql"],
                        "template":  tmpl.name,
                        "tip":       "clasic",
                        "descriere": result.get("descriere", tmpl.descriere),
                        "slots":     result.get("slots", {}),
                        "error":     None,
                    }

        return {
            "question":  question,
            "sql":       None,
            "template":  None,
            "tip":       None,
            "descriere": None,
            "slots":     {},
            "error":     "Niciun șablon potrivit găsit pentru această întrebare.",
        }

    def execute(self, sql: str) -> list[dict]:
        try:
            cursor = self.conn.execute(sql)
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception as e:
            return [{"error": str(e)}]

    def ask(self, question: str) -> dict:
        result = self.translate(question)
        if result["sql"]:
            result["rezultate"] = self.execute(result["sql"])
        else:
            result["rezultate"] = []
        return result

if __name__ == "__main__":
    translator = NL2SQLTranslator()

    questions_clasice = [
        ("CLASICE (regex)", [
            "Arată toți pacienții",
            "Câți medici sunt?",
            "Medici din Cluj-Napoca",
            "Medici de cardiologie",
            "Pacienți cu vârsta peste 50 ani",
            "Costul total al consultațiilor",
            "Consultații din ianuarie 2024",
        ])
    ]

    questions_stratica = [
        ("STRATICA TIP 1 — <atribut> al <obiect>", [
            "Care este specialitatea medicului Popa?",
            "Care este vârsta pacientului Ionescu?",
            "Care este diagnosticul pacientului Gheorghe?",
        ]),
        ("STRATICA TIP 2 — <atribut> al <obiect1> al <obiect2>", [
            "Consultațiile pacientului Ionescu",
            "Consultațiile medicului Popa",
            "Costul consultațiilor pacientului Stan",
        ]),
        ("STRATICA TIP 3 — <verb_acțiune> <obiect> <valoare>", [
            "Ce pacienti a consultat medicul Popa?",
            "Ce medic a tratat pacientul Ionescu?",
            "Cine a consultat pacientul Gheorghe?",
        ]),
    ]

    all_groups = questions_clasice + questions_stratica

    for group_name, questions in all_groups:
        print(f"\n{'='*70}")
        print(f"  {group_name}")
        print(f"{'='*70}")
        for q in questions:
            r = translator.ask(q)
            print(f"\n📝 {r['question']}")
            if r["sql"]:
                print(f"🔧 [{r['tip']}]")
                print(f"🗄️  {r['sql'].strip()}")
                rows = r["rezultate"]
                print(f"📊 {len(rows)} rând(uri)", end="")
                if rows and "error" not in rows[0]:
                    print(f" → {dict(rows[0])}")
                else:
                    print()
            else:
                print(f"❌ {r['error']}")