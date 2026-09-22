import re
from utils.schema import SCHEMA_INFO
from templates.nlp import normalize, extract_quoted_or_capitalized, extract_number

def resolve_table(text_norm: str) -> str | None:
    for tabel, info in SCHEMA_INFO["tabele"].items():
        for sin in info["sinonime"]:
            if normalize(sin) in text_norm:
                return tabel
    return None


def resolve_specialitate(text: str) -> str | None:
    text_n = normalize(text)
    SPEC_ALIASES = {
        "cardiolog": "cardiologie",
        "neurolog":  "neurologie",
        "ortoped":   "ortopedie",
        "pediatr":   "pediatrie",
    }
    for alias, spec in SPEC_ALIASES.items():
        if alias in text_n:
            return spec.capitalize()
    for spec in SCHEMA_INFO["specialitati"]:
        if normalize(spec) in text_n:
            return spec.capitalize()
    return None


def resolve_sex(text_norm: str) -> str | None:
    if any(w in text_norm for w in ["barbati", "barbat", "masculin", "baiat"]):
        return "M"
    if any(w in text_norm for w in ["femei", "femeie", "feminin", "fata"]):
        return "F"
    return None


def resolve_order_col(text_norm: str, table: str) -> str | None:
    col_hints = {
        "varsta":   "varsta",
        "cost":     "cost",
        "pret":     "cost",
        "data":     "data",
        "durata":   "durata_minute",
        "experienta": "experienta_ani",
        "ani":      "experienta_ani",
    }
    for hint, col in col_hints.items():
        if hint in text_norm:
            return col
    return None


def resolve_order_dir(text_norm: str) -> str:
    if any(w in text_norm for w in ["descrescator", "descrescatoare", "mare", "maxim", "recent"]):
        return "DESC"
    return "ASC"

class Template:
    def __init__(self, name, patterns, generate_fn, descriere=""):
        self.name = name
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.generate_fn = generate_fn
        self.descriere = descriere

    def match(self, text_norm: str):
        for pat in self.patterns:
            m = pat.search(text_norm)
            if m:
                return m
        return None

    def generate(self, text_original: str, text_norm: str, match) -> dict | None:
        return self.generate_fn(text_original, text_norm, match)

def _t1_listare(orig, norm, m):
    table = resolve_table(norm)
    if not table:
        return None
    return {
        "sql": f"SELECT * FROM {table};",
        "descriere": f"Listare completă din {table}",
        "slots": {"table": table},
    }

T1 = Template(
    name="listare_simpla",
    patterns=[
        r"\b(arata|afiseaza|listeaza|da.mi|vreau|gaseste|cauta)\b.*(toti|toate|totii|toti)\b",
        r"\b(arata|afiseaza|listeaza)\b(?!.*filtru)",
    ],
    generate_fn=_t1_listare,
    descriere="Arată toți [pacienții/medicii/consultațiile]",
)

def _t2_numarare(orig, norm, m):
    table = resolve_table(norm)
    if not table:
        return None
    return {
        "sql": f"SELECT COUNT(*) AS total FROM {table};",
        "descriere": f"Numărare înregistrări din {table}",
        "slots": {"table": table},
    }

T2 = Template(
    name="numarare",
    patterns=[
        r"\b(cati|cate|numarul|cate)\b.*(pacienti|medici|consultatii|bolnavi|doctori|vizite)",
        r"cate.*(sunt|exista|avem)",
    ],
    generate_fn=_t2_numarare,
    descriere="Câți/câte [pacienți/medici] sunt?",
)

def _t3_filtru_oras(orig, norm, m):
    table = resolve_table(norm)
    if not table:
        table = "pacienti"
    oras_m = re.search(r"(?i)\bdin\b\s+([A-Z\u0102\u00C2\u00CE\u015E\u0162][a-z\u0103\u00e2\u00ee\u015f\u0163]+(?:-[A-Z\u0102\u00C2\u00CE\u015E\u0162][a-z\u0103\u00e2\u00ee\u015f\u0163]+)*)", orig)
    oras = oras_m.group(1) if oras_m else extract_quoted_or_capitalized(orig)
    if not oras:
        return None
    return {
        "sql": f"SELECT * FROM {table} WHERE oras = '{oras}';",
        "descriere": f"Filtrare {table} după oraș = {oras}",
        "slots": {"table": table, "oras": oras},
    }

T3 = Template(
    name="filtru_oras",
    patterns=[
        r"\bdin\b.+\b(cluj|bucuresti|timisoara|iasi|brasov|[A-Z][a-z]+)\b",
        r"\bdin orasul\b",
        r"\bdin localitate\b",
    ],
    generate_fn=_t3_filtru_oras,
    descriere="[Pacienți/medici] din [Oraș]",
)

def _t4_filtru_spec(orig, norm, m):
    spec = resolve_specialitate(orig)
    if not spec:
        return None
    return {
        "sql": f"SELECT * FROM medici WHERE specialitate = '{spec}';",
        "descriere": f"Medici cu specialitate = {spec}",
        "slots": {"specialitate": spec},
    }

T4 = Template(
    name="filtru_specialitate",
    patterns=[
        r"\b(medici|doctori|specialist)\b.*(cardiolog|neurolog|ortoped|pediatr|cardiologie|neurologie|ortopedie|pediatrie)",
        r"\b(cardiolog|neurolog|ortoped|pediatr)(ie|i|ilor)?\b",
        r"specialitate[a]?\b.*(cardiolog|neurolog|ortoped|pediatr)",
    ],
    generate_fn=_t4_filtru_spec,
    descriere="Medici cu specialitatea [X]",
)

def _t5_filtru_varsta(orig, norm, m):
    numar = extract_number(orig)
    if numar is None:
        return None

    if any(w in norm for w in ["mai mare", "peste", "mai batran", "minim"]):
        op = ">"
    elif any(w in norm for w in ["mai mic", "sub", "mai tanar", "maxim"]):
        op = "<"
    elif any(w in norm for w in ["exact", "exact", "egal"]):
        op = "="
    else:
        op = "="

    return {
        "sql": f"SELECT * FROM pacienti WHERE varsta {op} {int(numar)};",
        "descriere": f"Pacienți cu vârsta {op} {int(numar)}",
        "slots": {"op": op, "varsta": int(numar)},
    }

T5 = Template(
    name="filtru_varsta",
    patterns=[
        r"varsta\b.*(mai mare|mai mic|peste|sub|exact|\d+)",
        r"\b(peste|sub|mai (mare|mic) de)\b.*\d+\s*(ani)?",
        r"\d+\s*ani",
    ],
    generate_fn=_t5_filtru_varsta,
    descriere="Pacienți cu vârsta [op] [N] ani",
)

def _t6_filtru_sex(orig, norm, m):
    sex = resolve_sex(norm)
    if not sex:
        return None
    return {
        "sql": f"SELECT * FROM pacienti WHERE sex = '{sex}';",
        "descriere": f"Pacienți de sex = {sex}",
        "slots": {"sex": sex},
    }

T6 = Template(
    name="filtru_sex",
    patterns=[
        r"\b(barbati|barbat|masculin)\b",
        r"\b(femei|femeie|feminin)\b",
        r"pacienti\b.*(barbati|femei)",
    ],
    generate_fn=_t6_filtru_sex,
    descriere="Pacienți [bărbați/femei]",
)

def _t7_consultatii_pacient(orig, norm, m):
    names = re.findall(r"\b([A-Z\u0102\u00C2\u00CE\u015E\u0162][a-z\u0103\u00e2\u00ee\u015f\u0163]{2,})\b", orig)
    skip = {"Consultațiile", "Consultații", "Vizitele", "Vizite", "Programările", "Arată", "Afișează"}
    names = [n for n in names if n not in skip]
    nume = names[-1] if names else None
    if not nume:
        return None
    return {
        "sql": (
            f"SELECT p.nume, p.prenume, c.data, c.diagnostic, c.cost "
            f"FROM consultatii c "
            f"JOIN pacienti p ON c.pacient_id = p.id "
            f"WHERE p.nume = '{nume}' OR p.prenume = '{nume}';"
        ),
        "descriere": f"Consultații ale pacientului {nume}",
        "slots": {"pacient": nume},
    }

T7 = Template(
    name="consultatii_pacient",
    patterns=[
        r"consultatii(le)?\b.*(pacient|lui|ei)\b",
        r"(vizitele|programarile)\b.*pacient",
        r"(ce|care)\b.*consultatii\b.*\b[A-Z]",
    ],
    generate_fn=_t7_consultatii_pacient,
    descriere="Consultațiile pacientului [Nume]",
)

def _t8_consultatii_medic(orig, norm, m):
    names = re.findall(r"\b([A-Z\u0102\u00C2\u00CE\u015E\u0162][a-z\u0103\u00e2\u00ee\u015f\u0163]{2,})\b", orig)
    skip = {"Consultațiile", "Consultații", "Vizitele", "Vizite", "Pacienții", "Pacienți"}
    names = [n for n in names if n not in skip]
    nume = names[-1] if names else None
    if not nume:
        return None
    return {
        "sql": (
            f"SELECT m.nume, m.prenume, c.data, c.diagnostic, p.nume AS pacient "
            f"FROM consultatii c "
            f"JOIN medici m ON c.medic_id = m.id "
            f"JOIN pacienti p ON c.pacient_id = p.id "
            f"WHERE m.nume = '{nume}' OR m.prenume = '{nume}';"
        ),
        "descriere": f"Consultațiile medicului {nume}",
        "slots": {"medic": nume},
    }

T8 = Template(
    name="consultatii_medic",
    patterns=[
        r"consultatii(le)?\b.*(medic|doctor)\b",
        r"(pacienti|pacientii)\b.*(medicul|doctorul)\b",
    ],
    generate_fn=_t8_consultatii_medic,
    descriere="Consultațiile medicului [Nume]",
)

def _t9_cost(orig, norm, m):
    if any(w in norm for w in ["total", "suma", "sum"]):
        agg, label = "SUM", "total_cost"
    elif any(w in norm for w in ["medie", "media", "avg"]):
        agg, label = "AVG", "cost_mediu"
    elif any(w in norm for w in ["maxim", "cel mai mare", "max"]):
        agg, label = "MAX", "cost_maxim"
    elif any(w in norm for w in ["minim", "cel mai mic", "min"]):
        agg, label = "MIN", "cost_minim"
    else:
        agg, label = "SUM", "total_cost"

    return {
        "sql": f"SELECT {agg}(cost) AS {label} FROM consultatii;",
        "descriere": f"Agregare {agg} pe costul consultațiilor",
        "slots": {"aggregare": agg},
    }

T9 = Template(
    name="agregare_cost",
    patterns=[
        r"\b(total|suma|medie|media|maxim|minim)\b.*(cost|pret|suma)",
        r"(cost|pret|costul)\b.*(total|mediu|maxim|minim|cel mai)",
        r"(costul total|suma totala|pretul total)",
        r"cat\b.*(costa|cost)",
    ],
    generate_fn=_t9_cost,
    descriere="[Total/Media/Max/Min] cost consultații",
)

def _t10_sortare(orig, norm, m):
    table = resolve_table(norm) or "pacienti"
    col = resolve_order_col(norm, table) or "id"
    direction = resolve_order_dir(norm)
    return {
        "sql": f"SELECT * FROM {table} ORDER BY {col} {direction};",
        "descriere": f"Sortare {table} după {col} {direction}",
        "slots": {"table": table, "col": col, "dir": direction},
    }

T10 = Template(
    name="sortare",
    patterns=[
        r"\b(ordonati|ordonate|sortati|sortate)\b",
        r"\border\b.*\b(varsta|cost|data|experienta)",
        r"\b(crescator|descrescator)\b",
    ],
    generate_fn=_t10_sortare,
    descriere="[Entități] ordonate după [coloană]",
)

def _t11_perioada(orig, norm, m):
    an = re.search(r"\b(202[0-9])\b", orig)
    luna = re.search(r"\b(ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)\b", norm)

    LUNI = {
        "ianuarie":"01","februarie":"02","martie":"03","aprilie":"04",
        "mai":"05","iunie":"06","iulie":"07","august":"08",
        "septembrie":"09","octombrie":"10","noiembrie":"11","decembrie":"12"
    }

    if an and luna:
        y, m_str = an.group(1), LUNI[luna.group(1)]
        return {
            "sql": f"SELECT * FROM consultatii WHERE data LIKE '{y}-{m_str}-%';",
            "descriere": f"Consultații din {luna.group(1)} {y}",
            "slots": {"an": y, "luna": m_str},
        }
    elif an:
        return {
            "sql": f"SELECT * FROM consultatii WHERE data LIKE '{an.group(1)}-%';",
            "descriere": f"Consultații din {an.group(1)}",
            "slots": {"an": an.group(1)},
        }
    return None

T11 = Template(
    name="filtru_perioada",
    patterns=[
        r"\bdin\b.*(202[0-9])",
        r"\b(ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)\b",
        r"\b(luna|anul)\b",
    ],
    generate_fn=_t11_perioada,
    descriere="Consultații din [lună/an]",
)

TEMPLATES: list[Template] = [
    T4,
    T7,
    T8,
    T9,
    T11,
    T5,
    T6,
    T3,
    T2,
    T10,
    T1,
]
