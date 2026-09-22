from __future__ import annotations
import re
from templates.nlp import normalize
from dataclasses import dataclass, field
from typing import Optional

from grammar.earley_parser import ParseNode, ParseResult

ENTITY_TO_TABLE = {
    "pacienti": "pacienti",
    "pacientii": "pacienti",
    "pacient": "pacienti",
    "medici": "medici",
    "medicii": "medici",
    "medic": "medici",
    "medicului": "medici",
    "consultatii": "consultatii",
    "consultatiile": "consultatii",
    "programari": "programari",
    "retete": "retete",
    "diagnostice": "diagnostice",
    "doctori": "medici",
    "spitale": "spitale",
    "clinici": "clinici",
}

ATTR_TO_COLUMN = {
    "varsta": "varsta",
    "specialitate": "specialitate",
    "specialitatea": "specialitate",
    "diagnostic": "diagnostic",
    "diagnosticul": "diagnostic",
    "cost": "cost",
    "costul": "cost",
    "data": "data",
    "durata": "durata_minute",
    "oras": "oras",
    "nume": "nume",
    "prenume": "prenume",
    "gen": "sex",
    "sex": "sex",
    "experienta": "experienta_ani",
    "salariu": "salariu",
    "adresa": "adresa",
    "telefon": "telefon",
    "email": "email",
    "id": "id",
}

ATTR_TO_TABLE = {
    "varsta": "pacienti",
    "specialitate": "medici",
    "diagnostic": "consultatii",
    "cost": "consultatii",
    "data": "consultatii",
    "durata_minute": "consultatii",
    "oras": "pacienti",
    "nume": None,
    "prenume": None,
    "sex": "pacienti",
    "experienta_ani": "medici",
    "salariu": "medici",
    "adresa": None,
    "telefon": None,
    "email": None,
    "id": None,
}

AGG_TO_FUNC = {
    "cati": "COUNT",
    "cate": "COUNT",
    "numarul": "COUNT",
    "numar": "COUNT",
    "totalul": "SUM",
    "suma": "SUM",
    "media": "AVG",
    "maximul": "MAX",
    "minimul": "MIN",
}

ORDER_WORD_DEFAULT_DIR = {
    "ordonati": "ASC",
    "ordonate": "ASC",
    "sortati": "ASC",
    "sortate": "ASC",
    "ordonat": "ASC",
    "sortat": "ASC",
    "afisati": "ASC",
    "afisate": "ASC",
}

ORDER_DIR_MAP = {
    "crescator": "ASC",
    "descrescator": "DESC",
}

CMP_SIMPLE_TO_SQL = {
    "peste": ">",
    "sub": "<",
    "exact": "=",
    "egal": "=",
    "=": "=",
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
}

CMP_COMPLEX_TO_SQL = {
    "mai_mare_decat": ">",
    "mai_mic_decat": "<",
    "egal_cu": "=",
    "cel_mai_mare": "MAX",
    "cea_mai_mare": "MAX",
    "cel_mai_mic": "MIN",
    "cea_mai_mica": "MIN",
}

SEX_MAP = {
    "barbati": "M",
    "masculin": "M",
    "femei": "F",
    "feminin": "F",
}

SPECIALITY_VALUES = {
    "cardiologie": "Cardiologie",
    "neurologie": "Neurologie",
    "ortopedie": "Ortopedie",
    "pediatrie": "Pediatrie",
}

@dataclass
class FilterIR:
    column: str
    operator: str
    value: str | int | float | None
    table: Optional[str] = None


@dataclass
class OrderIR:
    column: str
    direction: str = "ASC"
    table: Optional[str] = None


@dataclass
class QueryIR:
    intent: str = "select"
    target_table: Optional[str] = None
    target_column: Optional[str] = None
    select_all: bool = True
    aggregation: Optional[str] = None
    aggregation_column: Optional[str] = None
    filters: list[FilterIR] = field(default_factory=list)
    order_by: Optional[OrderIR] = None
    tables: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    text = text.lower().strip()
    repl = {
        "ă": "a", "â": "a", "î": "i",
        "ș": "s", "ş": "s", "ț": "t", "ţ": "t",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def first_leaf(node: ParseNode) -> Optional[str]:
    if node.word is not None:
        return node.word
    for child in node.children:
        val = first_leaf(child)
        if val is not None:
            return val
    return None


def all_leaf_words(node: ParseNode) -> list[str]:
    if node.word is not None:
        return [node.word]
    out = []
    for child in node.children:
        out.extend(all_leaf_words(child))
    return out


def first_node(node: ParseNode, label: str) -> Optional[ParseNode]:
    if node.label == label:
        return node
    for child in node.children:
        found = first_node(child, label)
        if found is not None:
            return found
    return None


def find_nodes(node: ParseNode, label: str) -> list[ParseNode]:
    result = []
    if node.label == label:
        result.append(node)
    for child in node.children:
        result.extend(find_nodes(child, label))
    return result

def map_entity_word_to_table(word: str) -> Optional[str]:
    return ENTITY_TO_TABLE.get(normalize(word))


def map_attribute_word(word: str) -> tuple[Optional[str], Optional[str]]:
    col = ATTR_TO_COLUMN.get(normalize(word))
    if not col:
        return None, None
    return ATTR_TO_TABLE.get(col), col


def infer_value_type(word: str) -> str | int | float:
    w = normalize(word)
    if w == "_number_":
        return 0
    try:
        if "." in w:
            return float(w)
        return int(w)
    except Exception:
        return word


def extract_target(tree: ParseNode) -> tuple[Optional[str], Optional[str], bool]:
    target = first_node(tree, "TARGET")
    if target is None:
        return None, None, True

    entity_ref = first_node(target, "ENTITY_REF")
    if entity_ref is not None:
        ent = first_node(entity_ref, "ENTITY")
        if ent is not None:
            ent_word = first_leaf(ent)
            if ent_word:
                tbl = map_entity_word_to_table(ent_word)
                attr_nodes = find_nodes(target, "ATTRIBUTE")
                if attr_nodes:
                    attr_word = first_leaf(attr_nodes[0])
                    if attr_word:
                        _, col = map_attribute_word(attr_word)
                        return tbl, col, False

    attr_nodes = find_nodes(target, "ATTRIBUTE")
    if attr_nodes:
        word = first_leaf(attr_nodes[0])
        if word:
            tbl, col = map_attribute_word(word)
            return tbl, col, False

    ent_nodes = find_nodes(target, "ENTITY")
    if ent_nodes:
        word = first_leaf(ent_nodes[0])
        if word:
            tbl = map_entity_word_to_table(word)
            return tbl, None, True

    return None, None, True


def extract_aggregation(tree: ParseNode) -> tuple[Optional[str], Optional[str]]:
    agg_expr = first_node(tree, "AGG_EXPR")
    if agg_expr is not None:
        agg_word_node = first_node(agg_expr, "AGG_WORD")
        if agg_word_node is not None:
            agg_word = first_leaf(agg_word_node)
            if agg_word:
                func = AGG_TO_FUNC.get(normalize(agg_word))
                if func:
                    attr_node = first_node(agg_expr, "ATTRIBUTE")
                    if attr_node is not None:
                        attr_word = first_leaf(attr_node)
                        if attr_word:
                            _, col = map_attribute_word(attr_word)
                            return func, col or "*"

                    ent_node = first_node(agg_expr, "ENTITY")
                    if ent_node is not None:
                        return func, "*"

                    target_node = first_node(agg_expr, "TARGET")
                    if target_node is not None:
                        attr_in_target = first_node(target_node, "ATTRIBUTE")
                        if attr_in_target is not None:
                            attr_word = first_leaf(attr_in_target)
                            if attr_word:
                                _, col = map_attribute_word(attr_word)
                                return func, col or "*"

                    return func, "*"

    agg_word_node = first_node(tree, "AGG_WORD")
    if agg_word_node is not None:
        agg_word = first_leaf(agg_word_node)
        if agg_word:
            func = AGG_TO_FUNC.get(normalize(agg_word))
            if func:
                target = first_node(tree, "TARGET")
                if target is not None:
                    attr_node = first_node(target, "ATTRIBUTE")
                    if attr_node is not None:
                        attr_word = first_leaf(attr_node)
                        if attr_word:
                            _, col = map_attribute_word(attr_word)
                            return func, col or "*"
                return func, "*"

    return None, None


def extract_order(tree: ParseNode) -> Optional[OrderIR]:
    order_clause = first_node(tree, "ORDER_CLAUSE")
    if order_clause is None:
        return None

    direction = "ASC"

    dir_node = first_node(order_clause, "ORDER_DIR")
    if dir_node is not None:
        dir_word = first_leaf(dir_node)
        if dir_word:
            direction = ORDER_DIR_MAP.get(normalize(dir_word), "ASC")
    else:
        word_node = first_node(order_clause, "ORDER_WORD")
        if word_node is not None:
            order_word = first_leaf(word_node)
            if order_word:
                direction = ORDER_WORD_DEFAULT_DIR.get(normalize(order_word), "ASC")

    order_pp = first_node(order_clause, "ORDER_PP")
    if order_pp is not None:
        attr_node = first_node(order_pp, "ATTRIBUTE")
        if attr_node is not None:
            attr_word = first_leaf(attr_node)
            if attr_word:
                tbl, col = map_attribute_word(attr_word)
                return OrderIR(column=col, direction=direction, table=tbl)

    return None


def extract_filter_pp(filter_pp: ParseNode) -> list[FilterIR]:
    filters: list[FilterIR] = []

    adp_loc = first_node(filter_pp, "ADP_LOC")
    adp_with = first_node(filter_pp, "ADP_WITH")
    value_node = first_node(filter_pp, "VALUE")

    attr_cond = first_node(filter_pp, "ATTR_CONDITION")
    if adp_with is not None and attr_cond is not None:
        filters.extend(extract_attr_condition(attr_cond))
        return filters

    date_filter = first_node(filter_pp, "DATE_FILTER")
    if adp_loc is not None and date_filter is not None:
        vals = all_leaf_words(date_filter)

        month_map = {
            "ianuarie": "01", "februarie": "02", "martie": "03", "aprilie": "04",
            "mai": "05", "iunie": "06", "iulie": "07", "august": "08",
            "septembrie": "09", "octombrie": "10", "noiembrie": "11", "decembrie": "12"
        }

        month = None
        year = None

        for v in vals:
            nv = normalize(v)
            if nv in month_map:
                month = month_map[nv]
            elif str(v).isdigit() and len(str(v)) == 4:
                year = str(v)

        if year and month:
            filters.append(FilterIR(
                column="data", operator="LIKE", value=f"{year}-{month}-%", table="consultatii"
            ))
            return filters
        if year:
            filters.append(FilterIR(
                column="data", operator="LIKE", value=f"{year}-%", table="consultatii"
            ))
            return filters

    if adp_loc is not None and value_node is not None:
        words = all_leaf_words(value_node)
        if words:
            val = words[-1]
            filters.append(FilterIR(column="oras", operator="=", value=val, table="pacienti"))
            return filters

    if adp_with is not None and value_node is not None:
        words = all_leaf_words(value_node)
        if words:
            val = normalize(words[-1])
            if val in SPECIALITY_VALUES:
                filters.append(FilterIR(
                    column="specialitate",
                    operator="=",
                    value=SPECIALITY_VALUES[val],
                    table="medici"
                ))
            else:
                filters.append(FilterIR(
                    column="diagnostic",
                    operator="=",
                    value=words[-1],
                    table="consultatii"
                ))
            return filters

    return filters


def extract_attr_condition(attr_cond: ParseNode) -> list[FilterIR]:
    filters: list[FilterIR] = []

    attr_node = first_node(attr_cond, "ATTRIBUTE")
    if attr_node is None:
        return filters

    attr_word = first_leaf(attr_node)
    if not attr_word:
        return filters

    tbl, col = map_attribute_word(attr_word)
    if not col:
        return filters

    adj_node = first_node(attr_cond, "ADJ")
    if adj_node is not None:
        adj_word = normalize(first_leaf(adj_node) or "")
        if adj_word in SEX_MAP:
            filters.append(FilterIR(column=col, operator="=", value=SEX_MAP[adj_word], table=tbl))
            return filters

    value_node = first_node(attr_cond, "VALUE")
    if value_node is not None:
        value_words = all_leaf_words(value_node)
        if value_words:
            value_word = value_words[-1]
            cmp_simple = first_node(attr_cond, "CMP_SIMPLE")
            if cmp_simple is not None:
                cmp_word = normalize(first_leaf(cmp_simple) or "")
                op = CMP_SIMPLE_TO_SQL.get(cmp_word, "=")
                filters.append(FilterIR(column=col, operator=op, value=value_word, table=tbl))
                return filters

            filters.append(FilterIR(column=col, operator="=", value=value_word, table=tbl))
            return filters

    number_node = first_node(attr_cond, "NUMBER")
    if number_node is not None:
        num_words = all_leaf_words(number_node)
        if num_words:
            raw = num_words[-1]
            value = infer_value_type(raw)
            cmp_simple = first_node(attr_cond, "CMP_SIMPLE")
            if cmp_simple is not None:
                cmp_word = normalize(first_leaf(cmp_simple) or "")
                op = CMP_SIMPLE_TO_SQL.get(cmp_word, "=")
                filters.append(FilterIR(column=col, operator=op, value=value, table=tbl))
                return filters

    cmp_complex = first_node(attr_cond, "CMP_COMPLEX")
    if cmp_complex is not None:
        leaf_words = all_leaf_words(cmp_complex)

        for w in leaf_words:
            op = CMP_COMPLEX_TO_SQL.get(normalize(w))
            if op in {"MAX", "MIN"}:
                filters.append(FilterIR(column=col, operator=op, value=None, table=tbl))
                return filters

        cmp_word = None
        for w in leaf_words:
            if normalize(w) in CMP_COMPLEX_TO_SQL:
                cmp_word = normalize(w)
                break

        op = CMP_COMPLEX_TO_SQL.get(cmp_word, "=") if cmp_word else "="

        number_node = first_node(cmp_complex, "NUMBER")
        if number_node is not None:
            vals = all_leaf_words(number_node)
            if vals:
                value = infer_value_type(vals[-1])
                filters.append(FilterIR(column=col, operator=op, value=value, table=tbl))
                return filters

        value_node = first_node(cmp_complex, "VALUE")
        if value_node is not None:
            vals = all_leaf_words(value_node)
            if vals:
                filters.append(FilterIR(column=col, operator=op, value=vals[-1], table=tbl))
                return filters

    return filters


def extract_filters(tree: ParseNode) -> list[FilterIR]:
    filters: list[FilterIR] = []

    for pp in find_nodes(tree, "FILTER_PP"):
        filters.extend(extract_filter_pp(pp))

    for cond in find_nodes(tree, "ATTR_CONDITION"):
        filters.extend(extract_attr_condition(cond))

    for entity_ref in find_nodes(tree, "ENTITY_REF"):
        ent = first_node(entity_ref, "ENTITY")
        val = first_node(entity_ref, "VALUE")
        if ent is not None and val is not None:
            ent_word = first_leaf(ent)
            val_word = first_leaf(val)
            if ent_word and val_word:
                tbl = map_entity_word_to_table(ent_word)
                if tbl in {"medici", "pacienti"}:
                    filters.append(FilterIR(
                        column="nume",
                        operator="=",
                        value=val_word,
                        table=tbl
                    ))

    return filters

def tree_to_ir(result: ParseResult) -> QueryIR | None:
    if not result.success or result.tree is None:
        return None

    tree = result.tree
    ir = QueryIR()

    target_table, target_column, select_all = extract_target(tree)
    ir.target_table = target_table
    ir.target_column = target_column
    ir.select_all = select_all

    agg_func, agg_col = extract_aggregation(tree)
    if agg_func:
        ir.intent = "aggregate"
        ir.aggregation = agg_func
        ir.aggregation_column = agg_col
    else:
        ir.intent = "select"

    ir.order_by = extract_order(tree)
    ir.filters = extract_filters(tree)

    tables = set()

    if ir.target_table:
        tables.add(ir.target_table)

    if ir.order_by and ir.order_by.table:
        tables.add(ir.order_by.table)

    for f in ir.filters:
        if f.table:
            tables.add(f.table)

    if not tables and ir.target_column:
        tbl = ATTR_TO_TABLE.get(ir.target_column)
        if tbl:
            tables.add(tbl)

    ir.tables = list(tables)

    if ir.target_table is None and ir.tables:
        ir.target_table = ir.tables[0]

    return ir

def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def build_where(filters: list[FilterIR]) -> tuple[list[str], Optional[tuple[str, str]]]:
    where = []
    extremum = None

    for f in filters:
        col_ref = f"{f.table}.{f.column}" if f.table else f.column

        if f.operator in {"MAX", "MIN"}:
            extremum = (f.operator, col_ref)
            continue

        where.append(f"{col_ref} {f.operator} {sql_literal(f.value)}")

    return where, extremum


def build_from_and_joins(target_table: Optional[str], tables: list[str]) -> tuple[str, list[str]]:
    tables = list(dict.fromkeys(tables))

    if target_table and target_table not in tables:
        tables.insert(0, target_table)

    if not tables:
        return "", []

    base = target_table or tables[0]
    remaining = [t for t in tables if t != base]

    from_clause = base
    join_conds = []

    if base == "pacienti" and "consultatii" in remaining:
        from_clause = "pacienti JOIN consultatii ON consultatii.pacient_id = pacienti.id"
        remaining.remove("consultatii")

    elif base == "consultatii" and "pacienti" in remaining:
        from_clause = "consultatii JOIN pacienti ON consultatii.pacient_id = pacienti.id"
        remaining.remove("pacienti")


    if "medici" in remaining and "consultatii" in from_clause:
        from_clause += " JOIN medici ON consultatii.medic_id = medici.id"
        remaining.remove("medici")

    elif base == "medici" and "consultatii" in remaining:
        from_clause = "medici JOIN consultatii ON consultatii.medic_id = medici.id"
        remaining.remove("consultatii")

    for t in remaining:
        from_clause += f", {t}"

    return from_clause, join_conds


def ir_to_sql(ir: QueryIR) -> dict:
    if ir is None:
        return {"sql": None, "error": "IR inexistent"}

    if not ir.tables and not ir.target_table:
        return {"sql": None, "error": "Nicio tabelă identificată"}

    tables = ir.tables[:] if ir.tables else ([ir.target_table] if ir.target_table else [])
    from_clause, join_conds = build_from_and_joins(ir.target_table, tables)
    where_conds, extremum = build_where(ir.filters)
    where_conds = join_conds + where_conds

    if ir.aggregation:
        col = ir.aggregation_column or "*"
        if col == "*":
            select_clause = f"{ir.aggregation}(*) AS rezultat"
        else:
            col_ref = col
            if ATTR_TO_TABLE.get(col):
                col_ref = f"{ATTR_TO_TABLE[col]}.{col}"
            select_clause = f"{ir.aggregation}({col_ref}) AS rezultat"

    elif ir.target_column:
        col_ref = ir.target_column
        if ir.target_table:
            col_ref = f"{ir.target_table}.{ir.target_column}"
        select_clause = col_ref

    else:
        if ir.target_table:
            select_clause = f"{ir.target_table}.*"
        else:
            select_clause = "*"

    sql = f"SELECT {select_clause} FROM {from_clause}"

    if extremum is not None:
        op, col_ref = extremum
        where_conds.append(
            f"{col_ref} = (SELECT {op}({col_ref}) FROM {from_clause})"
        )

    if where_conds:
        sql += " WHERE " + " AND ".join(where_conds)

    if ir.order_by is not None:
        col_ref = ir.order_by.column
        if ir.order_by.table:
            col_ref = f"{ir.order_by.table}.{ir.order_by.column}"
        sql += f" ORDER BY {col_ref} {ir.order_by.direction}"

    sql += ";"

    return {
        "sql": sql,
        "ir": ir,
        "error": None,
    }

def tree_to_sql(result: ParseResult) -> dict:
    if not result.success or result.tree is None:
        return {
            "sql": None,
            "ir": None,
            "error": result.error,
            "prob": result.probability,
        }

    ir = tree_to_ir(result)
    out = ir_to_sql(ir)
    out["prob"] = result.probability
    return out

_ORIGINAL_TREE_TO_SQL = tree_to_sql

_CITY_DISPLAY = {
    "cluj-napoca": "Cluj-Napoca", "cluj": "Cluj-Napoca", "bucuresti": "Bucuresti",
    "iasi": "Iasi", "timisoara": "Timisoara", "constanta": "Constanta", "sibiu": "Sibiu",
    "oradea": "Oradea", "brasov": "Brasov", "craiova": "Craiova", "bacau": "Bacau",
    "zalau": "Zalau", "resita": "Resita", "suceava": "Suceava", "arad": "Arad",
    "pitesti": "Pitesti", "ploiesti": "Ploiesti", "buzau": "Buzau", "deva": "Deva",
    "tulcea": "Tulcea", "focsani": "Focsani", "galati": "Galati", "alba_iulia": "Alba Iulia",
    "baia_mare": "Baia Mare", "satu_mare": "Satu Mare", "sfantu_gheorghe": "Sfantu Gheorghe",
    "piatra_neamt": "Piatra Neamt", "ramnicu_valcea": "Ramnicu Valcea",
    "drobeta_turnu_severin": "Drobeta-Turnu Severin", "targu_mures": "Targu Mures",
}

_SPEC_DISPLAY = {
    "cardiologie": "Cardiologie", "cardiologi": "Cardiologie", "cardiolog": "Cardiologie",
    "neurologie": "Neurologie", "neurologi": "Neurologie", "neurolog": "Neurologie",
    "ortopedie": "Ortopedie", "ortopezi": "Ortopedie", "ortoped": "Ortopedie",
    "pediatrie": "Pediatrie", "pediatri": "Pediatrie", "pediatru": "Pediatrie",
    "urologie": "Urologie", "urologi": "Urologie", "urolog": "Urologie",
    "nefrologie": "Nefrologie", "nefrologi": "Nefrologie", "nefrolog": "Nefrologie",
    "dermatologie": "Dermatologie", "dermatologi": "Dermatologie", "dermatolog": "Dermatologie",
    "psihiatrie": "Psihiatrie", "psihiatri": "Psihiatrie", "psihiatru": "Psihiatrie",
    "gastroenterologie": "Gastroenterologie", "gastroenterologi": "Gastroenterologie", "gastroenterolog": "Gastroenterologie",
    "oncologie": "Oncologie", "oncologi": "Oncologie", "oncolog": "Oncologie",
    "oftalmologie": "Oftalmologie", "oftalmologi": "Oftalmologie", "oftalmolog": "Oftalmologie",
    "hematologie": "Hematologie", "hematologi": "Hematologie", "hematolog": "Hematologie",
    "pneumologie": "Pneumologie", "pneumologi": "Pneumologie", "pneumolog": "Pneumologie",
    "endocrinologie": "Endocrinologie", "endocrinologi": "Endocrinologie", "endocrinolog": "Endocrinologie",
    "reumatologie": "Reumatologie", "reumatologi": "Reumatologie", "reumatolog": "Reumatologie",
    "chirurgie": "Chirurgie", "chirurgi": "Chirurgie", "chirurg": "Chirurgie",
}

_MONTHS = {
    "ianuarie": "01", "februarie": "02", "martie": "03", "aprilie": "04", "mai": "05",
    "iunie": "06", "iulie": "07", "august": "08", "septembrie": "09", "octombrie": "10",
    "noiembrie": "11", "decembrie": "12",
}

_NAME_STOP = {
    "cine", "care", "ce", "unde", "din", "este", "sunt", "arata", "afiseaza", "listeaza",
    "prezinta", "da", "vreau", "pacientul", "pacientului", "medicul", "medicului", "doctorul",
    "doctorului", "consultațiile", "consultatiile", "vizitele", "istoricul", "medical", "curant",
}


def _norm_q(text: str) -> str:
    text = normalize(text)
    repl = {
        "da-mi": "da", "da mi": "da", "arata-mi": "arata", "arata mi": "arata",
        "afiseaza-mi": "afiseaza", "prezinta-mi": "prezinta",
        "mai mare decat": "mai_mare_decat", "mai mare de": "mai_mare_decat",
        "mai mult decat": "mai_mare_decat", "mai mult de": "mai_mare_decat",
        "mai batrani de": "mai_mare_decat", "mai batran de": "mai_mare_decat",
        "in varsta de peste": "peste", "mai mic decat": "mai_mic_decat", "mai mic de": "mai_mic_decat",
        "mai mica de": "mai_mic_decat", "mai mici de": "mai_mic_decat", "mai putin de": "mai_mic_decat",
        "mai tineri de": "mai_mic_decat", "mai tanar de": "mai_mic_decat",
        "mai scurte de": "mai_mic_decat", "mai lungi de": "mai_mare_decat",
        "egal cu": "egal_cu", "de sex masculin": "barbati", "de sex feminin": "femei",
        "alba iulia": "alba_iulia", "baia mare": "baia_mare", "satu mare": "satu_mare",
        "sfantu gheorghe": "sfantu_gheorghe", "targu mures": "targu_mures", "piatra neamt": "piatra_neamt",
        "ramnicu valcea": "ramnicu_valcea", "drobeta-turnu severin": "drobeta_turnu_severin",
        "drobeta turnu severin": "drobeta_turnu_severin",
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    text = re.sub(r"[^a-z0-9_><= -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_city(q: str) -> str | None:
    for key, val in sorted(_CITY_DISPLAY.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(key)}\b", q):
            return val
    return None


def _first_spec(q: str) -> str | None:
    for key, val in sorted(_SPEC_DISPLAY.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(key)}\b", q):
            return val
    return None


def _first_number(q: str) -> str | None:
    m = re.search(r"\b(\d+(?:[.,]\d+)?)\b", q)
    return m.group(1).replace(",", ".") if m else None


def _cmp(q: str) -> str | None:
    if any(x in q for x in ["mai_mic_decat", " sub ", "sub ", "mai scurte", "mai ieftin"]):
        return "<"
    if any(x in q for x in ["mai_mare_decat", " peste ", "peste ", "mai lungi", "mai scumpe"]):
        return ">"
    if any(x in q for x in ["egal_cu", " exact ", "="]):
        return "="
    return None


def _name_after(q: str, markers: list[str]) -> str | None:
    for marker in markers:
        m = re.search(rf"\b{marker}\s+([a-z][a-z_\-]+)\b", q)
        if m:
            name = m.group(1)
            if name not in _NAME_STOP and name not in _CITY_DISPLAY and name not in _SPEC_DISPLAY:
                return name.capitalize()
    return None


def _where(parts: list[str]) -> str:
    return " WHERE " + " AND ".join(parts) if parts else ""


def _rule_based_sql(question: str) -> str | None:
    q = _norm_q(question)
    city = _first_city(q)
    spec = _first_spec(q)
    num = _first_number(q)
    op = _cmp(q)
    is_count = bool(re.search(r"\b(cati|cate|numarul|numar|totalul)\b", q)) and not ("cost" in q and "total" in q)

    is_medici = any(w in q for w in ["medici", "medicii", "medic ", "medicul", "doctor", "doctori"])
    is_pacienti = any(w in q for w in ["pacienti", "pacientii", "pacient ", "pacientul", "pacientului", "bolnavi", "paciente", "pacientele"])
    is_cons = any(w in q for w in ["consultatii", "consultatiile", "vizite", "vizitele"])

    patient_name = _name_after(q, ["pacientul", "pacientului", "pacient"])
    doctor_name = _name_after(q, ["medicul", "medicului", "doctorul", "doctorului"])
    relation_words = ["tratat", "consultat", "vazut", "curant", "ocupat", "beneficiat", "medicul pacientului"]
    if patient_name and any(w in q for w in relation_words):
        return ("SELECT m.nume, m.prenume, m.specialitate, c.data "
                "FROM pacienti p JOIN consultatii c ON c.pacient_id = p.id "
                "JOIN medici m ON c.medic_id = m.id "
                f"WHERE p.nume = '{patient_name}' OR p.prenume = '{patient_name}';")

    if is_cons and patient_name:
        return ("SELECT p.nume, p.prenume, c.data, c.diagnostic, c.cost "
                "FROM pacienti p JOIN consultatii c ON c.pacient_id = p.id "
                f"WHERE p.nume = '{patient_name}' OR p.prenume = '{patient_name}';")
    if is_cons and doctor_name:
        return ("SELECT m.nume, m.prenume, c.data, c.diagnostic "
                "FROM medici m JOIN consultatii c ON c.medic_id = m.id "
                f"WHERE m.nume = '{doctor_name}' OR m.prenume = '{doctor_name}';")

    attr_patterns = [
        ("telefon", "telefon", "pacienti", patient_name),
        ("varsta", "varsta", "pacienti", patient_name),
        ("sex", "sex", "pacienti", patient_name),
        ("oras", "oras", "pacienti", patient_name),
        ("specialitate", "specialitate", "medici", doctor_name),
        ("experienta", "experienta_ani", "medici", doctor_name),
    ]
    if "ani are pacientul" in q and patient_name:
        return f"SELECT varsta FROM pacienti WHERE nume = '{patient_name}' OR prenume = '{patient_name}';"
    if "ani lucreaza medicul" in q and doctor_name:
        return f"SELECT experienta_ani FROM medici WHERE nume = '{doctor_name}' OR prenume = '{doctor_name}';"
    for token, col, table, name in attr_patterns:
        if token in q and name:
            return f"SELECT {col} FROM {table} WHERE nume = '{name}' OR prenume = '{name}';"

    year = None
    ym = re.search(r"\b(20\d{2})\b", q)
    if ym:
        year = ym.group(1)
    month = next((m for m in _MONTHS if re.search(rf"\b{m}\b", q)), None)
    if is_cons and year:
        pattern = f"{year}-{_MONTHS[month]}-%" if month else f"{year}-%"
        select = "COUNT(*) AS total" if is_count else "*"
        return f"SELECT {select} FROM consultatii WHERE data LIKE '{pattern}';"

    if "cost" in q and any(w in q for w in ["total", "suma", "costurilor"]):
        return "SELECT SUM(cost) AS total_cost FROM consultatii;"
    if "experienta" in q and "medie" in q:
        return "SELECT AVG(experienta_ani) AS medie_exp FROM medici;"
    if "varsta" in q and ("maxim" in q or "cea_mai_mare" in q):
        return "SELECT MAX(varsta) AS max_varsta FROM pacienti;"
    if "varsta" in q and ("minim" in q or "cea_mai_mica" in q):
        return "SELECT MIN(varsta) AS min_varsta FROM pacienti;"
    if "cost" in q and ("ieftin" in q or "minim" in q):
        return "SELECT MIN(cost) AS min_cost FROM consultatii;"

    conditions = []
    target_table = "medici" if is_medici and not is_pacienti else "pacienti" if is_pacienti else "consultatii" if is_cons else None
    if spec:
        conditions.append(f"specialitate = '{spec}'")
        target_table = "medici"
    if city:
        conditions.append(f"oras = '{city}'")
    if any(w in q for w in ["barbati", "masculin"]):
        conditions.append("sex = 'M'")
        target_table = "pacienti"
    if any(w in q for w in ["femei", "feminin", "paciente", "pacientele"]):
        conditions.append("sex = 'F'")
        target_table = "pacienti"
    if num and op:
        if "experienta" in q:
            conditions.append(f"experienta_ani {op} {num}")
            target_table = "medici"
        elif "durata" in q or "minute" in q:
            conditions.append(f"durata_minute {op} {num}")
            target_table = "consultatii"
        elif "cost" in q or "lei" in q:
            conditions.append(f"cost {op} {num}")
            target_table = "consultatii"
        else:
            conditions.append(f"varsta {op} {num}")
            target_table = "pacienti"

    if target_table:
        select = "COUNT(*) AS total" if is_count else "*"
        return f"SELECT {select} FROM {target_table}{_where(conditions)};"

    return None


def tree_to_sql(result: ParseResult) -> dict:
    fallback_sql = _rule_based_sql(result.question)
    if fallback_sql:
        return {
            "sql": fallback_sql,
            "ir": None,
            "error": None,
            "prob": result.probability,
            "fallback": True,
        }

    out = _ORIGINAL_TREE_TO_SQL(result)

    sql = out.get("sql") if isinstance(out, dict) else None
    if sql:
        if "FROM medici" in sql:
            sql = sql.replace("pacienti.oras", "medici.oras")
        elif "FROM pacienti" in sql:
            sql = sql.replace("medici.oras", "pacienti.oras")
        out["sql"] = sql
    return out
