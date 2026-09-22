from __future__ import annotations
import re
import unicodedata
import json
from grammar.pcfg import MEDICAL_GRAMMAR
from grammar.earley_parser import ProbabilisticEarleyParser
from grammar.tree_to_sql import tree_to_sql
from utils.schema import get_connection, build_schema_info



class GrammarNL2SQL:
    def __init__(self, db_conn=None):
        self.grammar = MEDICAL_GRAMMAR
        self.parser = ProbabilisticEarleyParser(self.grammar)

        if db_conn is None:
            self.conn = get_connection()
            build_schema_info(self.conn)
        else:
            self.conn = db_conn

    def translate(self, question: str, verbose: bool = False) -> dict:
        parse_result = self.parser.parse(question)

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"Întrebare: {question}")
            print(f"Tokeni: {parse_result.tokens}")
            print(
                f"Parsare: {'✓' if parse_result.success else '✗'} "
                f"(prob={parse_result.probability:.6f})"
            )
            if parse_result.tree:
                print("\n   Arbore de parsare:")
                print(parse_result.tree.pretty_print(indent=2))
            if parse_result.error:
                print(f"\n   Eroare parsare: {parse_result.error}")

        result = tree_to_sql(parse_result)
        result["question"] = question
        result["parse_success"] = parse_result.success
        result["tokens"] = parse_result.tokens

        if verbose:
            if result.get("ir") is not None:
                print("\n   IR:")
                print(f"     {result['ir']}")
            print(f"\n   SQL: {result.get('sql', 'N/A')}")
            if result.get("error"):
                print(f"   Eroare SQL: {result['error']}")

        return result

    def ask(self, question: str, verbose: bool = False) -> dict:
        result = self.translate(question, verbose=verbose)

        sql = result.get("sql")
        if sql:
            result["rezultate"] = self._execute(sql)
        else:
            result["rezultate"] = []

        return result

    def _execute(self, sql: str) -> list[dict]:
        try:
            cursor = self.conn.execute(sql)
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception as e:
            return [{"error": str(e)}]

def load_test_cases(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        (item["question"], item["sql"])
        for item in data
        if "question" in item and "sql" in item
    ]

def run_evaluation(json_path: str = "test.json"):
    import json
    from templates.translator import NL2SQLTranslator

    def load_test_cases(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            (item["question"], item["sql"])
            for item in data
            if "question" in item and "sql" in item
        ]

    def strip_diacritics(text: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def normalize_sql(sql):
        sql = strip_diacritics(sql)

        sql = sql.lower().strip()

        sql = re.sub(r"\s+", " ", sql)

        sql = re.sub(r"\s*\(\s*", "(", sql)
        sql = re.sub(r"\s*\)\s*", ")", sql)

        sql = re.sub(r"\s*,\s*", ", ", sql)

        sql = re.sub(r"\s*;\s*", ";", sql)

        return sql

    def is_exec_ok(rows):
        return not rows or "error" not in rows[0]

    grammar_sys = GrammarNL2SQL()
    template_sys = NL2SQLTranslator()

    test_cases = load_test_cases(json_path)

    print("\n" + "=" * 95)
    print("  COMPARAȚIE: Grammar-based (PCFG + Earley) vs Template-based")
    print("=" * 95)
    print(
        f"{'Întrebare':<40} "
        f"{'G_Parse':^8} {'G_Exec':^8} {'G_Exact':^8} "
        f"{'T_Exec':^8} {'T_Exact':^8}"
    )
    print("─" * 95)

    grammar_parse_ok = 0
    grammar_sql_ok = 0
    grammar_exec_ok = 0
    grammar_exact_ok = 0

    template_exec_ok = 0
    template_exact_ok = 0

    error_stats = {
        "sql_generation_fail": 0,
        "execution_fail": 0,
        "fallback_mismatch": 0,
        "exact_mismatch": 0,
        "ok": 0,
    }

    for question, expected in test_cases:
        g_res = grammar_sys.translate(question)
        t_res = template_sys.translate(question)

        g_sql = g_res.get("sql")
        t_sql = t_res.get("sql")

        g_rows = grammar_sys._execute(g_sql) if g_sql else []
        t_rows = template_sys.execute(t_sql) if t_sql else []

        g_parse = bool(g_res.get("parse_success"))
        g_sql_generated = g_sql is not None
        g_exec = g_sql_generated and is_exec_ok(g_rows)
        g_exact = g_sql is not None and normalize_sql(g_sql) == normalize_sql(expected)

        if not g_sql_generated:
            g_reason = "sql_generation_fail"
        elif not g_exec:
            g_reason = "execution_fail"
        elif g_exact:
            g_reason = "ok"
        elif not g_parse:
            g_reason = "fallback_mismatch"
        else:
            g_reason = "exact_mismatch"

        error_stats[g_reason] += 1

        t_exec = t_sql is not None and is_exec_ok(t_rows)
        t_exact = t_sql is not None and normalize_sql(t_sql) == normalize_sql(expected)

        if g_parse:
            grammar_parse_ok += 1
        if g_sql_generated:
            grammar_sql_ok += 1
        if g_exec:
            grammar_exec_ok += 1
        if g_exact:
            grammar_exact_ok += 1

        if t_exec:
            template_exec_ok += 1
        if t_exact:
            template_exact_ok += 1

        q_short = question[:38] + ".." if len(question) > 38 else question

        print(
            f"{q_short:<40} "
            f"{'✅' if g_parse else '❌':^8} "
            f"{'✅' if g_exec else '❌':^8} "
            f"{'✅' if g_exact else '❌':^8} "
            f"{'✅' if t_exec else '❌':^8} "
            f"{'✅' if t_exact else '❌':^8}"
        )

        if g_reason != "ok":
            print(f"  Grammar reason: {g_reason}")
            print(f"  Grammar SQL: {g_sql or 'N/A'}")
            print(f"  Expected SQL: {expected}")

    total = len(test_cases)

    print("─" * 95)

    print("\n  REZULTATE GRAMMAR-BASED")
    print(f"  Parse Success Rate:       {grammar_parse_ok}/{total} ({grammar_parse_ok / total:.0%})")
    print(f"  SQL Generation Rate:      {grammar_sql_ok}/{total} ({grammar_sql_ok / total:.0%})")
    print(f"  Execution Success Rate:   {grammar_exec_ok}/{total} ({grammar_exec_ok / total:.0%})")
    print(f"  Exact Match Rate:         {grammar_exact_ok}/{total} ({grammar_exact_ok / total:.0%})")

    print("\n  REZULTATE TEMPLATE-BASED")
    print(f"  Execution Success Rate:   {template_exec_ok}/{total} ({template_exec_ok / total:.0%})")
    print(f"  Exact Match Rate:         {template_exact_ok}/{total} ({template_exact_ok / total:.0%})")

    print("\n  DISTRIBUȚIE ERORI GRAMMAR-BASED")
    for reason, count in error_stats.items():
        print(f"  {reason:<25} {count}/{total} ({count / total:.0%})")

    print("=" * 95)


if __name__ == "__main__":
    system = GrammarNL2SQL()

    demo_questions = [
        "Arată toți pacienții",
        "Câți medici sunt?",
        "Medici din Cluj-Napoca",
        "Medici de cardiologie",
        "Pacienți cu vârsta peste 50",
        "Arată pacienții bărbați",
        "Costul total al consultațiilor",
        "Pacienți ordonați descrescător după vârstă",
        "Consultații din ianuarie 2024",
        "Care este specialitatea medicului Popa?",
    ]

    print("=" * 60)
    print("  Grammar-based NL2SQL — PCFG + Earley")
    print("=" * 60)

    ok = 0
    for q in demo_questions:
        r = system.ask(q)
        sql = r.get("sql")
        rows = r.get("rezultate", [])

        exec_ok = sql and (not rows or "error" not in rows[0])
        if exec_ok:
            ok += 1

        icon = "✅" if exec_ok else "❌"
        print(f"\n{icon} {q}")
        print(f"   SQL: {sql or 'N/A'}")
        print(f"   Rezultate: {len(rows)} rânduri | P(parse)={r.get('prob', 0):.6f}")

    print(f"\n{'─' * 60}")
    print(f"  Succes: {ok}/{len(demo_questions)} ({ok/len(demo_questions):.0%})")
    print(f"{'─' * 60}")

    print("\n\n" + "=" * 60)
    print("  DEMO VERBOSE — arbore de parsare vizibil")
    print("=" * 60)
    system.ask("Câți medici sunt?", verbose=True)

    print("\n")
    run_evaluation()