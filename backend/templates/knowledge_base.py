from templates.nlp import normalize

DEFAULT_ATTRIBUTES = {
    "pacienti":    ["nume", "prenume"],
    "medici":      ["nume", "prenume", "specialitate"],
    "consultatii": ["data", "diagnostic", "cost"],
}

ACTION_VERBS = {
    "consulta":       ("medici",   "pacienti"),
    "consultata":     ("medici",   "pacienti"),
    "consultat":      ("medici",   "pacienti"),
    "consulte":       ("medici",   "pacienti"),
    "trateaza":       ("medici",   "pacienti"),
    "tratat":         ("medici",   "pacienti"),
    "tratata":        ("medici",   "pacienti"),
    "vede":           ("medici",   "pacienti"),
    "vazut":          ("medici",   "pacienti"),
    "programeaza":    ("pacienti", "consultatii"),
    "programat":      ("pacienti", "consultatii"),
    "programata":     ("pacienti", "consultatii"),
    "interneaza":     ("pacienti", "consultatii"),
    "internat":       ("pacienti", "consultatii"),
    "internata":      ("pacienti", "consultatii"),

    "lucreaza":       ("medici",   None),
    "activeaza":      ("medici",   None),
    "specializa":     ("medici",   None),

    "sufera":         ("pacienti", None),
    "diagnosticat":   ("pacienti", None),
    "diagnosticata":  ("pacienti", None),
}

RELATION_RULES = {
    frozenset(["pacienti", "consultatii"]): {
        "tables":    ["pacienti p", "consultatii c"],
        "join_cond": "c.pacient_id = p.id",
        "select_default": "p.nume, p.prenume, c.data, c.diagnostic, c.cost",
    },

    frozenset(["medici", "consultatii"]): {
        "tables":    ["medici m", "consultatii c"],
        "join_cond": "c.medic_id = m.id",
        "select_default": "m.nume, m.prenume, m.specialitate, c.data, c.diagnostic",
    },

    frozenset(["pacienti", "medici", "consultatii"]): {
        "tables":    ["pacienti p", "medici m", "consultatii c"],
        "join_cond": "c.pacient_id = p.id AND c.medic_id = m.id",
        "select_default": "p.nume AS pacient, m.nume AS medic, m.specialitate, c.data, c.diagnostic, c.cost",
    },
}


def get_relation_rule(tables: list[str]) -> dict | None:
    key = frozenset(tables)
    return RELATION_RULES.get(key)


def build_join_sql(tables: list[str], where_extra: str = "", select_override: str = "") -> str | None:
    rule = get_relation_rule(tables)
    if not rule:
        return None

    select = select_override or rule["select_default"]
    from_clause = ", ".join(rule["tables"])
    where = rule["join_cond"]
    if where_extra:
        where += f" AND ({where_extra})"

    return f"SELECT {select} FROM {from_clause} WHERE {where};"

SEMANTIC_SETS = {
    "pacient":      ("pacienti",    None),
    "pacienti":     ("pacienti",    None),
    "bolnav":       ("pacienti",    None),
    "bolnavi":      ("pacienti",    None),
    "persoana":     ("pacienti",    None),
    "persoane":     ("pacienti",    None),
    "nume":         (None,          "nume"),
    "prenume":      (None,          "prenume"),
    "varsta":       ("pacienti",    "varsta"),
    "varste":       ("pacienti",    "varsta"),
    "varste":       ("pacienti",    "varsta"),
    "sex":          ("pacienti",    "sex"),
    "gen":          ("pacienti",    "sex"),
    "oras":         (None,          "oras"),
    "localitate":   (None,          "oras"),
    "telefon":      ("pacienti",    "telefon"),

    "medic":        ("medici",      None),
    "medici":       ("medici",      None),
    "doctor":       ("medici",      None),
    "doctori":      ("medici",      None),
    "specialist":   ("medici",      None),
    "specialisti":  ("medici",      None),
    "specialitate": ("medici",      "specialitate"),
    "experienta":   ("medici",      "experienta_ani"),

    "consultatie":  ("consultatii", None),
    "consultatii":  ("consultatii", None),
    "vizita":       ("consultatii", None),
    "vizite":       ("consultatii", None),
    "programare":   ("consultatii", None),
    "programari":   ("consultatii", None),
    "diagnostic":   ("consultatii", "diagnostic"),
    "diagnostice":  ("consultatii", "diagnostic"),
    "cost":         ("consultatii", "cost"),
    "pret":         ("consultatii", "cost"),
    "suma":         ("consultatii", "cost"),
    "data":         ("consultatii", "data"),
    "durata":       ("consultatii", "durata_minute"),
}


def resolve_semantic(term: str) -> tuple[str | None, str | None]:
    return SEMANTIC_SETS.get(normalize(term), (None, None))


def resolve_action_verb(text_norm: str) -> tuple[str, str] | None:
    for verb, tables in ACTION_VERBS.items():
        if verb in text_norm:
            return tables
    return None