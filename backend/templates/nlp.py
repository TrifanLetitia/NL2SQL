import re
import unicodedata

def normalize(text: str) -> str:
    text = text.lower().strip()
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

LEMMA_MAP = {
    "pacientii": "pacienti",
    "pacientilor": "pacienti",
    "pacientul": "pacient",
    "pacientei": "pacient",
    "bolnavii": "bolnavi",
    "bolnavilor": "bolnavi",
    "bolnavul": "bolnav",
    "persoanele": "persoane",
    "persoanelor": "persoane",

    "medicii": "medici",
    "medicilor": "medici",
    "medicul": "medic",
    "doctorii": "doctori",
    "doctorilor": "doctori",
    "doctorul": "doctor",
    "specialistii": "specialisti",
    "specialistilor": "specialisti",
    "consultatiil": "consultatii",
    "consultatiilor": "consultatii",
    "consultatia": "consultatie",
    "vizitele": "vizite",
    "vizitelor": "vizite",
    "programarile": "programari",
    "programarilor": "programari",
    "arata": "arata",
    "afiseaza": "arata",
    "listeaza": "arata",
    "da-mi": "da",
    "dami": "da",
    "vreau": "vreau",
    "gaseste": "gaseste",
    "cauta": "gaseste",
}

def lemmatize(token: str) -> str:
    t = normalize(token)
    return LEMMA_MAP.get(t, t)


def tokenize(text: str) -> list[str]:
    text = normalize(text)
    tokens = re.findall(r"\b\w+\b", text)
    return [lemmatize(t) for t in tokens]

FILTER_KEYWORDS = {
    "din":      "oras",
    "din orasul": "oras",
    "varsta":   "varsta",
    "ani":      "varsta",
    "specialist": "specialitate",
    "specialitatea": "specialitate",
    "specialitate":  "specialitate",
    "diagnostic":    "diagnostic",
    "cu diagnosticul": "diagnostic",
}

COMPARISON_MAP = {
    "mai mare":    ">",
    "mai mare de": ">",
    "mai mic":     "<",
    "mai mic de":  "<",
    "cel putin":   ">=",
    "cel mult":    "<=",
    "egal":        "=",
    "exact":       "=",
}

AGGREGATION_KEYWORDS = {
    "cati":   "COUNT",
    "cate":   "COUNT",
    "numarul": "COUNT",
    "suma":   "SUM",
    "total":  "SUM",
    "media":  "AVG",
    "medie":  "AVG",
    "maxim":  "MAX",
    "minim":  "MIN",
    "cel mai mare": "MAX",
    "cel mai mic":  "MIN",
}

ORDER_KEYWORDS = {
    "ordonate": True,
    "ordonati": True,
    "sortate":  True,
    "sortati":  True,
    "crescator": "ASC",
    "descrescator": "DESC",
    "descrescatoare": "DESC",
    "crescatoare": "ASC",
}

def extract_number(text: str) -> int | float | None:
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if m:
        val = m.group(1)
        return float(val) if "." in val else int(val)
    return None


def extract_quoted_or_capitalized(text: str) -> str | None:
    m = re.search(r'["\']([^"\']+)["\']', text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-ZĂÂÎȘȚ][a-zăâîșț]+(?:-[A-ZĂÂÎȘȚ][a-zăâîșț]+)*)\b", text)
    if m:
        return m.group(1)
    return None
