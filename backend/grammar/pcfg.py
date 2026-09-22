from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass(frozen=True)
class PCFGRule:
    lhs: str
    rhs: tuple[str, ...]
    prob: float
    is_lexical: bool = False

    def __repr__(self) -> str:
        return f"{self.lhs} -> {' '.join(self.rhs)} [{self.prob:.3f}]"


class MedicalPCFG:
    def __init__(self):
        self.rules: list[PCFGRule] = []
        self._by_lhs: dict[str, list[PCFGRule]] = defaultdict(list)
        self._by_word: dict[str, list[PCFGRule]] = defaultdict(list)

        self._build_grammar()
        self._validate()

    def _add(self, lhs: str, rhs: list[str], prob: float, is_lexical: bool = False):
        rule = PCFGRule(lhs=lhs, rhs=tuple(rhs), prob=prob, is_lexical=is_lexical)
        self.rules.append(rule)
        self._by_lhs[lhs].append(rule)

        if is_lexical:
            self._by_word[rhs[0].lower()].append(rule)

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        repl = {
            "ă": "a",
            "â": "a",
            "î": "i",
            "ș": "s",
            "ş": "s",
            "ț": "t",
            "ţ": "t",
        }
        for k, v in repl.items():
            text = text.replace(k, v)
        text = re.sub(r"\s+", " ", text)
        return text

    def _build_grammar(self):

        self._add("S", ["SELECT_QUERY"], 0.45)
        self._add("S", ["AGG_QUERY"], 0.25)
        self._add("S", ["ORDER_QUERY"], 0.10)
        self._add("S", ["ELLIPTIC_QUERY"], 0.20)

        # SELECT
        self._add("SELECT_QUERY", ["SELECT_VERB", "TARGET"], 0.25)
        self._add("SELECT_QUERY", ["SELECT_VERB", "TARGET", "FILTER"], 0.30)
        self._add("SELECT_QUERY", ["QUESTION_WORD", "COPULA", "TARGET"], 0.15)
        self._add("SELECT_QUERY", ["QUESTION_WORD", "COPULA", "TARGET", "FILTER"], 0.20)
        self._add("SELECT_QUERY", ["QUESTION_WORD", "TARGET"], 0.10)

        # AGG
        self._add("AGG_QUERY", ["AGG_WORD", "TARGET"], 0.20)
        self._add("AGG_QUERY", ["AGG_WORD", "TARGET", "COPULA"], 0.15)
        self._add("AGG_QUERY", ["AGG_WORD", "TARGET", "FILTER"], 0.20)
        self._add("AGG_QUERY", ["QUESTION_WORD", "COPULA", "AGG_EXPR"], 0.15)
        self._add("AGG_QUERY", ["QUESTION_WORD", "COPULA", "AGG_EXPR", "FILTER"], 0.15)
        self._add("AGG_QUERY", ["AGG_EXPR"], 0.15)

        # ORDER
        self._add("ORDER_QUERY", ["SELECT_VERB", "TARGET", "ORDER_CLAUSE"], 0.40)
        self._add("ORDER_QUERY", ["SELECT_VERB", "TARGET", "FILTER", "ORDER_CLAUSE"], 0.35)
        self._add("ORDER_QUERY", ["TARGET", "ORDER_CLAUSE"], 0.15)
        self._add("ORDER_QUERY", ["TARGET", "FILTER", "ORDER_CLAUSE"], 0.10)

        # ELLIPTIC
        self._add("ELLIPTIC_QUERY", ["TARGET"], 0.45)
        self._add("ELLIPTIC_QUERY", ["TARGET", "FILTER"], 0.35)
        self._add("ELLIPTIC_QUERY", ["AGG_WORD", "TARGET"], 0.20)

        self._add("TARGET", ["ENTITY"], 0.26)
        self._add("TARGET", ["ATTRIBUTE"], 0.14)
        self._add("TARGET", ["DET", "ENTITY"], 0.08)
        self._add("TARGET", ["ENTITY", "FILTER_PP"], 0.10)
        self._add("TARGET", ["ENTITY", "ADJ"], 0.10)
        self._add("TARGET", ["ATTRIBUTE", "OF_ENTITY"], 0.14)
        self._add("TARGET", ["ATTRIBUTE", "ADJ"], 0.10)
        self._add("TARGET", ["ATTRIBUTE", "ENTITY_REF"], 0.08)

        self._add("ENTITY_REF", ["ENTITY", "VALUE"], 0.70)
        self._add("ENTITY_REF", ["DET", "ENTITY", "VALUE"], 0.30)


        self._add("AGG_EXPR", ["AGG_WORD", "ENTITY"], 0.25)
        self._add("AGG_EXPR", ["AGG_WORD", "ATTRIBUTE"], 0.25)
        self._add("AGG_EXPR", ["AGG_WORD", "ATTRIBUTE", "OF_ENTITY"], 0.20)
        self._add("AGG_EXPR", ["AGG_WORD", "DET", "ENTITY"], 0.15)
        self._add("AGG_EXPR", ["AGG_WORD", "TARGET"], 0.15)

        self._add("FILTER", ["FILTER_PP"], 0.25)
        self._add("FILTER", ["ATTR_CONDITION"], 0.35)
        self._add("FILTER", ["FILTER_PP", "FILTER_PP"], 0.10)
        self._add("FILTER", ["FILTER_PP", "ATTR_CONDITION"], 0.10)
        self._add("FILTER", ["ATTR_CONDITION", "FILTER_PP"], 0.10)
        self._add("FILTER", ["ATTR_CONDITION", "CONJ", "ATTR_CONDITION"], 0.10)

        self._add("ORDER_CLAUSE", ["ORDER_WORD"], 0.08)
        self._add("ORDER_CLAUSE", ["ORDER_DIR"], 0.08)
        self._add("ORDER_CLAUSE", ["ORDER_WORD", "ORDER_PP"], 0.30)
        self._add("ORDER_CLAUSE", ["ORDER_WORD", "ORDER_DIR"], 0.14)
        self._add("ORDER_CLAUSE", ["ORDER_WORD", "ORDER_PP", "ORDER_DIR"], 0.10)
        self._add("ORDER_CLAUSE", ["ORDER_WORD", "ORDER_DIR", "ORDER_PP"], 0.20)
        self._add("ORDER_CLAUSE", ["ORDER_DIR", "ORDER_PP"], 0.10)

        self._add("OF_ENTITY", ["ADP_OF", "ENTITY"], 0.60)
        self._add("OF_ENTITY", ["ADP_OF", "DET", "ENTITY"], 0.25)
        self._add("OF_ENTITY", ["ADP_OF", "VALUE"], 0.15)

        self._add("FILTER_PP", ["ADP_LOC", "VALUE"], 0.22)
        self._add("FILTER_PP", ["ADP_WITH", "VALUE"], 0.13)
        self._add("FILTER_PP", ["ADP_WITH", "ATTRIBUTE"], 0.07)
        self._add("FILTER_PP", ["ADP_WITH", "ATTR_CONDITION"], 0.13)
        self._add("FILTER_PP", ["ADP_OF", "ATTRIBUTE"], 0.10)
        self._add("FILTER_PP", ["ADP_OF", "VALUE"], 0.08)
        self._add("FILTER_PP", ["ADP_LOC", "ENTITY"], 0.07)
        self._add("FILTER_PP", ["ADP_WITH", "ENTITY"], 0.10)
        self._add("FILTER_PP", ["ADP_LOC", "DATE_FILTER"], 0.10)


        self._add("ATTR_CONDITION", ["ATTRIBUTE", "CMP_SIMPLE", "VALUE"], 0.35)
        self._add("ATTR_CONDITION", ["ATTRIBUTE", "CMP_SIMPLE", "NUMBER"], 0.25)
        self._add("ATTR_CONDITION", ["ATTRIBUTE", "CMP_COMPLEX"], 0.15)
        self._add("ATTR_CONDITION", ["ATTRIBUTE", "VALUE"], 0.10)
        self._add("ATTR_CONDITION", ["ATTRIBUTE", "ADJ"], 0.15)

        self._add("CMP_COMPLEX", ["CMP_GT", "NUMBER"], 0.35)
        self._add("CMP_COMPLEX", ["CMP_LT", "NUMBER"], 0.25)
        self._add("CMP_COMPLEX", ["CMP_EQ", "VALUE"], 0.20)
        self._add("CMP_COMPLEX", ["CMP_EQ", "NUMBER"], 0.10)
        self._add("CMP_COMPLEX", ["CMP_SUP"], 0.05)
        self._add("CMP_COMPLEX", ["CMP_INF"], 0.05)

        self._add("ORDER_PP", ["ADP_ORDER", "ATTRIBUTE"], 0.80)
        self._add("ORDER_PP", ["ADP_ORDER", "ENTITY"], 0.20)

        self._add("DATE_FILTER", ["VALUE", "NUMBER"], 0.60)
        self._add("DATE_FILTER", ["NUMBER"], 0.20)
        self._add("DATE_FILTER", ["VALUE"], 0.20)

        self._add("QUESTION_WORD", ["care"], 0.35, True)
        self._add("QUESTION_WORD", ["ce"], 0.20, True)
        self._add("QUESTION_WORD", ["cine"], 0.10, True)
        self._add("QUESTION_WORD", ["care_sunt"], 0.15, True)
        self._add("QUESTION_WORD", ["care_este"], 0.10, True)
        self._add("QUESTION_WORD", ["ce_este"], 0.10, True)

        self._add("SELECT_VERB", ["arata"], 0.30, True)
        self._add("SELECT_VERB", ["afiseaza"], 0.20, True)
        self._add("SELECT_VERB", ["listeaza"], 0.20, True)
        self._add("SELECT_VERB", ["gaseste"], 0.15, True)
        self._add("SELECT_VERB", ["da"], 0.10, True)
        self._add("SELECT_VERB", ["selecteaza"], 0.05, True)

        self._add("COPULA", ["este"], 0.45, True)
        self._add("COPULA", ["sunt"], 0.55, True)

        self._add("ENTITY", ["pacienti"], 0.17, True)
        self._add("ENTITY", ["pacientii"], 0.08, True)
        self._add("ENTITY", ["pacient"], 0.05, True)
        self._add("ENTITY", ["medici"], 0.14, True)
        self._add("ENTITY", ["medicii"], 0.06, True)
        self._add("ENTITY", ["medic"], 0.05, True)
        self._add("ENTITY", ["consultatii"], 0.11, True)
        self._add("ENTITY", ["consultatiile"], 0.05, True)
        self._add("ENTITY", ["programari"], 0.08, True)
        self._add("ENTITY", ["retete"], 0.06, True)
        self._add("ENTITY", ["diagnostice"], 0.05, True)
        self._add("ENTITY", ["doctori"], 0.04, True)
        self._add("ENTITY", ["spitale"], 0.04, True)
        self._add("ENTITY", ["clinici"], 0.02, True)

        self._add("ATTRIBUTE", ["varsta"], 0.15, True)
        self._add("ATTRIBUTE", ["specialitate"], 0.08, True)
        self._add("ATTRIBUTE", ["specialitatea"], 0.07, True)
        self._add("ATTRIBUTE", ["diagnostic"], 0.10, True)
        self._add("ATTRIBUTE", ["diagnosticul"], 0.05, True)
        self._add("ATTRIBUTE", ["cost"], 0.08, True)
        self._add("ATTRIBUTE", ["costul"], 0.06, True)
        self._add("ATTRIBUTE", ["data"], 0.05, True)
        self._add("ATTRIBUTE", ["durata"], 0.05, True)
        self._add("ATTRIBUTE", ["oras"], 0.05, True)
        self._add("ATTRIBUTE", ["nume"], 0.04, True)
        self._add("ATTRIBUTE", ["prenume"], 0.03, True)
        self._add("ATTRIBUTE", ["gen"], 0.03, True)
        self._add("ATTRIBUTE", ["sex"], 0.03, True)
        self._add("ATTRIBUTE", ["experienta"], 0.03, True)
        self._add("ATTRIBUTE", ["salariu"], 0.03, True)
        self._add("ATTRIBUTE", ["adresa"], 0.02, True)
        self._add("ATTRIBUTE", ["telefon"], 0.02, True)
        self._add("ATTRIBUTE", ["email"], 0.02, True)
        self._add("ATTRIBUTE", ["id"], 0.01, True)

        self._add("DET", ["toti"], 0.22, True)
        self._add("DET", ["toate"], 0.20, True)
        self._add("DET", ["cei"], 0.08, True)
        self._add("DET", ["cele"], 0.08, True)
        self._add("DET", ["un"], 0.12, True)
        self._add("DET", ["o"], 0.12, True)
        self._add("DET", ["niste"], 0.08, True)
        self._add("DET", ["toti_pacientii"], 0.05, True)
        self._add("DET", ["toti_medicii"], 0.05, True)

        self._add("ADJ", ["total"], 0.18, True)
        self._add("ADJ", ["mediu"], 0.12, True)
        self._add("ADJ", ["maxim"], 0.10, True)
        self._add("ADJ", ["minim"], 0.08, True)
        self._add("ADJ", ["barbati"], 0.10, True)
        self._add("ADJ", ["femei"], 0.10, True)
        self._add("ADJ", ["masculin"], 0.07, True)
        self._add("ADJ", ["feminin"], 0.07, True)
        self._add("ADJ", ["activ"], 0.05, True)
        self._add("ADJ", ["internat"], 0.05, True)
        self._add("ADJ", ["externat"], 0.03, True)
        self._add("ADJ", ["urgent"], 0.03, True)
        self._add("ADJ", ["recent"], 0.02, True)

        self._add("AGG_WORD", ["cati"], 0.18, True)
        self._add("AGG_WORD", ["cate"], 0.14, True)
        self._add("AGG_WORD", ["numarul"], 0.16, True)
        self._add("AGG_WORD", ["totalul"], 0.12, True)
        self._add("AGG_WORD", ["media"], 0.12, True)
        self._add("AGG_WORD", ["suma"], 0.08, True)
        self._add("AGG_WORD", ["maximul"], 0.08, True)
        self._add("AGG_WORD", ["minimul"], 0.07, True)
        self._add("AGG_WORD", ["numar"], 0.05, True)

        self._add("ORDER_WORD", ["ordonati"], 0.18, True)
        self._add("ORDER_WORD", ["ordonate"], 0.12, True)
        self._add("ORDER_WORD", ["sortati"], 0.18, True)
        self._add("ORDER_WORD", ["sortate"], 0.12, True)
        self._add("ORDER_WORD", ["ordonat"], 0.15, True)
        self._add("ORDER_WORD", ["sortat"], 0.15, True)
        self._add("ORDER_WORD", ["afisati"], 0.05, True)
        self._add("ORDER_WORD", ["afisate"], 0.05, True)

        self._add("ORDER_DIR", ["crescator"], 0.45, True)
        self._add("ORDER_DIR", ["descrescator"], 0.55, True)

        self._add("ADP_LOC", ["din"], 0.60, True)
        self._add("ADP_LOC", ["la"], 0.25, True)
        self._add("ADP_LOC", ["in"], 0.15, True)

        self._add("ADP_WITH", ["cu"], 0.70, True)
        self._add("ADP_WITH", ["pentru"], 0.20, True)
        self._add("ADP_WITH", ["avand"], 0.10, True)

        self._add("ADP_OF", ["de"], 0.35, True)
        self._add("ADP_OF", ["al"], 0.20, True)
        self._add("ADP_OF", ["a"], 0.15, True)
        self._add("ADP_OF", ["ale"], 0.12, True)
        self._add("ADP_OF", ["ai"], 0.08, True)
        self._add("ADP_OF", ["ale_lui"], 0.10, True)

        self._add("ADP_ORDER", ["dupa"], 1.00, True)

        self._add("CMP_SIMPLE", ["peste"], 0.25, True)
        self._add("CMP_SIMPLE", ["sub"], 0.20, True)
        self._add("CMP_SIMPLE", ["exact"], 0.10, True)
        self._add("CMP_SIMPLE", ["egal"], 0.05, True)
        self._add("CMP_SIMPLE", ["="], 0.05, True)
        self._add("CMP_SIMPLE", [">"], 0.15, True)
        self._add("CMP_SIMPLE", ["<"], 0.10, True)
        self._add("CMP_SIMPLE", [">="], 0.05, True)
        self._add("CMP_SIMPLE", ["<="], 0.05, True)

        self._add("CMP_GT", ["mai_mare_decat"], 1.00, True)
        self._add("CMP_LT", ["mai_mic_decat"], 1.00, True)
        self._add("CMP_EQ", ["egal_cu"], 1.00, True)

        self._add("CMP_SUP", ["cel_mai_mare"], 0.60, True)
        self._add("CMP_SUP", ["cea_mai_mare"], 0.40, True)

        self._add("CMP_INF", ["cel_mai_mic"], 0.60, True)
        self._add("CMP_INF", ["cea_mai_mica"], 0.40, True)

        self._add("CONJ", ["si"], 0.75, True)
        self._add("CONJ", ["sau"], 0.25, True)

        self._add("VALUE", ["_CITY_"], 0.20, True)
        self._add("VALUE", ["_DIAG_"], 0.15, True)
        self._add("VALUE", ["_PROPN_"], 0.20, True)
        self._add("VALUE", ["_NUMBER_"], 0.10, True)
        self._add("VALUE", ["_DATE_"], 0.10, True)
        self._add("VALUE", ["_TEXT_"], 0.10, True)
        self._add("VALUE", ["cardiologie"], 0.05, True)
        self._add("VALUE", ["neurologie"], 0.03, True)
        self._add("VALUE", ["ortopedie"], 0.02, True)
        self._add("VALUE", ["pediatrie"], 0.02, True)
        self._add("VALUE", ["diabet"], 0.01, True)
        self._add("VALUE", ["hipertensiune"], 0.01, True)
        self._add("VALUE", ["covid"], 0.01, True)

        self._add("NUMBER", ["_NUMBER_"], 1.00, True)

    def _validate(self):
        totals = defaultdict(float)
        for rule in self.rules:
            totals[rule.lhs] += rule.prob

        bad = []
        for lhs, total in totals.items():
            if abs(total - 1.0) > 1e-9:
                bad.append((lhs, total))

        if bad:
            msg = "\n".join(f"{lhs}: {total:.6f}" for lhs, total in bad)
            raise ValueError(
                "Gramatica nu este un PCFG valid. Sumele pe LHS nu dau 1.0:\n" + msg
            )

    def get_rules_for(self, lhs: str) -> list[PCFGRule]:
        return self._by_lhs.get(lhs, [])

    def get_lex_rules_for(self, word: str) -> list[PCFGRule]:
        word = self.normalize(word)
        rules = list(self._by_word.get(word, []))
        rules.extend(self._dynamic_rules_for(word))
        return rules

    def _dynamic_rules_for(self, word: str) -> list[PCFGRule]:
        rules: list[PCFGRule] = []

        if re.fullmatch(r"\d+([.,]\d+)?", word):
            rules.append(PCFGRule("VALUE", ("_NUMBER_",), 0.10, True))
            rules.append(PCFGRule("NUMBER", ("_NUMBER_",), 1.00, True))

        cities = {
            "cluj", "cluj-napoca", "bucuresti", "iasi", "timisoara",
            "constanta", "sibiu", "oradea", "brasov", "craiova"
        }
        if word in cities:
            rules.append(PCFGRule("VALUE", ("_CITY_",), 0.20, True))

        diags = {
            "diabet", "hipertensiune", "gripa", "astm", "covid",
            "anemie", "aritmie", "migrena", "fractura"
        }
        if word in diags:
            rules.append(PCFGRule("VALUE", ("_DIAG_",), 0.15, True))

        dates = {
            "ianuarie", "februarie", "martie", "aprilie", "mai",
            "iunie", "iulie", "august", "septembrie", "octombrie",
            "noiembrie", "decembrie"
        }
        if word in dates:
            rules.append(PCFGRule("VALUE", ("_DATE_",), 0.10, True))

        if re.fullmatch(r"[a-z][a-z-]+", word) and word not in cities and word not in diags and word not in dates:
            rules.append(PCFGRule("VALUE", ("_PROPN_",), 0.03, True))

        return rules


    def print_grammar(self):
        print("PCFG medicala pentru Earley")
        print("=" * 60)
        current = None
        for rule in sorted(self.rules, key=lambda r: (r.lhs, -r.prob, r.rhs)):
            if rule.lhs != current:
                current = rule.lhs
                print(f"\n{current}:")
            print(f"  -> {' '.join(rule.rhs):<35} [{rule.prob:.3f}]")


MEDICAL_GRAMMAR = MedicalPCFG()