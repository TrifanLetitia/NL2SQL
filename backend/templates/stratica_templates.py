import re

from utils.schema import SCHEMA_INFO
from templates.nlp import normalize
from templates.knowledge_base import build_join_sql, resolve_action_verb, ACTION_VERBS
from templates.knowledge_base import DEFAULT_ATTRIBUTES

def extract_name(orig: str, skip_words: set = None) -> str | None:
    skip = skip_words or set()
    names = re.findall(
        r"\b([A-ZĂÂÎȘȚ][a-zăâîșț]{2,}(?:-[A-ZĂÂÎȘȚ][a-zăâîșț]+)*)\b", orig
    )
    names = [n for n in names if n not in skip]
    return names[-1] if names else None


def resolve_table_from_text(norm: str) -> str | None:
    for tabel, info in SCHEMA_INFO["tabele"].items():
        for sin in info["sinonime"]:
            if normalize(sin) in norm:
                return tabel
    return None


def resolve_column_from_text(norm: str) -> str | None:
    COLUMN_HINTS = {
        "specialitate":  "specialitate",
        "specialitatea": "specialitate",
        "varsta":        "varsta",
        "varste":        "varsta",
        "diagnostic":    "diagnostic",
        "diagnosticul":  "diagnostic",
        "cost":          "cost",
        "costul":        "cost",
        "pret":          "cost",
        "data":          "data",
        "durata":        "durata_minute",
        "experienta":    "experienta_ani",
        "telefon":       "telefon",
        "sex":           "sex",
        "gen":           "sex",
        "oras":          "oras",
        "orasul":        "oras",
        "localitate":    "oras",
        "prenume":       "prenume",
        "nume":          "nume",
    }
    for hint, col in COLUMN_HINTS.items():
        if hint in norm:
            return col
    return None

TIP1_PATTERNS = [
    re.compile(r"\b(specialitatea|varsta|costul?|diagnosticul?|durata|telefon|prenumele?|orasul?|experienta)\b.*(medicului?|pacientului?|doctorului?)", re.I),
    re.compile(r"\b(care este|care-i|ce este|ce)\b.*(specialitatea|varsta|diagnosticul?|costul?)\b", re.I),
    re.compile(r"\b(specialitatea|varsta|diagnosticul?|costul?|orasul?|experienta)\b.*\b(lui|ei|sa|pentru)\b", re.I),
]

def match_tip1(orig: str, norm: str) -> dict | None:
    for pat in TIP1_PATTERNS:
        if pat.search(norm):
            col = resolve_column_from_text(norm)
            table = resolve_table_from_text(norm)

            if not col and not table:
                continue

            if not table:
                col_to_table = {
                    "specialitate": "medici", "experienta_ani": "medici",
                    "varsta": "pacienti", "sex": "pacienti", "telefon": "pacienti",
                    "diagnostic": "consultatii", "cost": "consultatii", "durata_minute": "consultatii",
                    "oras": None,
                }
                table = col_to_table.get(col)

            if not table:
                continue

            skip = {"Care", "Este", "Cine", "Care", "Medicului", "Pacientului"}
            name = extract_name(orig, skip)

            select_col = col or ", ".join(DEFAULT_ATTRIBUTES.get(table, ["*"]))

            if name:
                sql = f"SELECT {select_col} FROM {table} WHERE nume = '{name}' OR prenume = '{name}';"
            else:
                sql = f"SELECT {select_col} FROM {table};"

            return {
                "tip":      "TIP1_atribut_of_obiect",
                "sql":      sql,
                "descriere": f"[Tip 1] {select_col} din {table}" + (f" unde nume='{name}'" if name else ""),
                "slots":    {"col": select_col, "table": table, "filtru": name},
            }
    return None

TIP2_PATTERNS = [
    re.compile(r"\b(diagnosticul?|costul?|data|durata)\b.*(pacientului?|medicului?)", re.I),
    re.compile(r"\b(consultatiile?|vizitele?)\b.*(pacientului?|medicului?|doctorului?)", re.I),
    re.compile(r"\b(medicul?|doctorul?)\b.*(pacientului?|bolnavului?)", re.I),
    re.compile(r"\b(pacientii?|bolnavii?)\b.*(medicului?|doctorului?)", re.I),
    re.compile(r"\b(consultat|tratat|vazut)\b.*(de catre|de)\b", re.I),
]

def match_tip2(orig: str, norm: str) -> dict | None:
    for pat in TIP2_PATTERNS:
        if not pat.search(norm):
            continue

        col = resolve_column_from_text(norm)
        name = extract_name(orig, {"Diagnosticul", "Costul", "Consultatiile",
                                    "Medicul", "Pacientii", "Vizitele"})

        has_pacient = any(w in norm for w in ["pacient", "bolnav"])
        has_medic   = any(w in norm for w in ["medic", "doctor", "specialist"])
        has_consult = any(w in norm for w in ["consultati", "vizit", "diagnostic", "cost", "durata"])

        if has_pacient and has_consult and not has_medic:
            tables = ["pacienti", "consultatii"]
        elif has_medic and has_consult and not has_pacient:
            tables = ["medici", "consultatii"]
        elif has_pacient and has_medic:
            tables = ["pacienti", "medici", "consultatii"]
        else:
            continue

        where_extra = ""
        if name:
            if has_pacient and not has_medic:
                where_extra = f"p.nume = '{name}' OR p.prenume = '{name}'"
            elif has_medic and not has_pacient:
                where_extra = f"m.nume = '{name}' OR m.prenume = '{name}'"
            else:
                where_extra = f"(p.nume = '{name}' OR m.nume = '{name}')"

        select_override = ""
        if col:
            prefix_map = {
                "pacienti": "p", "medici": "m", "consultatii": "c"
            }
            col_table = {
                "diagnostic": "c", "cost": "c", "data": "c", "durata_minute": "c",
                "specialitate": "m", "experienta_ani": "m",
                "varsta": "p", "sex": "p", "telefon": "p",
                "nume": None, "prenume": None, "oras": None,
            }
            prefix = col_table.get(col, "")
            select_override = f"{prefix}.{col}" if prefix else col

        sql = build_join_sql(tables, where_extra, select_override)
        if not sql:
            continue

        return {
            "tip":      "TIP2_atribut_of_obiect_of_obiect",
            "sql":      sql,
            "descriere": f"[Tip 2] JOIN {'+'.join(tables)}" + (f" filtru='{name}'" if name else ""),
            "slots":    {"tables": tables, "col": col, "filtru": name},
        }
    return None

TIP3_PATTERNS = [
    re.compile(r"\b(consultat[a]?|tratat[a]?|vazut[a]?|programat[a]?|internat[a]?)\b", re.I),
    re.compile(r"\b(cine|care medic|ce medic)\b.*(consultat?|tratat?|programat?)", re.I),
    re.compile(r"\b(ce pacienti?|cine)\b.*(consultat?|tratat?|internat?)", re.I),
]

def match_tip3(orig: str, norm: str) -> dict | None:
    for pat in TIP3_PATTERNS:
        if not pat.search(norm):
            continue

        verb_tables = resolve_action_verb(norm)
        if not verb_tables:
            for verb in ["consultat", "tratat", "vazut", "programat", "internat"]:
                if verb in norm:
                    verb_tables = ACTION_VERBS.get(verb)
                    break

        if not verb_tables:
            continue

        name = extract_name(orig, {"Cine", "Care", "Pacienti", "Medic", "Doctor"})

        cauta_medic   = any(w in norm for w in ["cine", "care medic", "ce medic", "medicul"])
        cauta_pacient = any(w in norm for w in ["ce pacienti", "ce pacient", "cine a fost"])

        tables = ["pacienti", "medici", "consultatii"]

        if name:
            if cauta_medic:
                where_extra = f"p.nume = '{name}' OR p.prenume = '{name}'"
                select_override = "m.nume AS medic, m.prenume, m.specialitate, c.data, c.diagnostic"
            elif cauta_pacient:
                where_extra = f"m.nume = '{name}' OR m.prenume = '{name}'"
                select_override = "p.nume AS pacient, p.prenume, c.data, c.diagnostic"
            else:
                where_extra = f"(p.nume = '{name}' OR m.nume = '{name}')"
                select_override = ""
        else:
            where_extra = ""
            select_override = ""

        sql = build_join_sql(tables, where_extra, select_override)
        if not sql:
            continue

        return {
            "tip":      "TIP3_verb_actiune",
            "sql":      sql,
            "descriere": f"[Tip 3] Verb acțiune → JOIN {'+'.join(tables)}" + (f" filtru='{name}'" if name else ""),
            "slots":    {"tables": tables, "verb": "detectat", "filtru": name},
        }
    return None


def match_stratica_templates(orig: str, norm: str) -> dict | None:
    return match_tip3(orig, norm) or match_tip2(orig, norm) or match_tip1(orig, norm)