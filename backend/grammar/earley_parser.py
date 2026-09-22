from __future__ import annotations
from dataclasses import dataclass, field
import math
import re
from collections import defaultdict, deque

@dataclass
class ParseNode:
    label: str
    prob: float
    word: str | None = None
    children: list["ParseNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.word is not None

    def pretty_print(self, indent: int = 0) -> str:
        pad = "  " * indent
        if self.is_leaf:
            return f"{pad}[{self.label}: '{self.word}' | p={math.exp(self.prob):.6f}]"
        body = "\n".join(child.pretty_print(indent + 1) for child in self.children)
        return f"{pad}({self.label} | p={math.exp(self.prob):.6f}\n{body}\n{pad})"


@dataclass
class ParseResult:
    question: str
    tokens: list[str]
    success: bool
    tree: ParseNode | None
    probability: float
    error: str = ""

@dataclass(frozen=True)
class EarleyKey:
    lhs: str
    rhs: tuple[str, ...]
    dot: int
    start: int
    end: int


@dataclass
class EarleyState:
    lhs: str
    rhs: tuple[str, ...]
    dot: int
    start: int
    end: int
    log_prob: float
    children: list[ParseNode] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.dot >= len(self.rhs)

    @property
    def next_symbol(self) -> str | None:
        if self.is_complete:
            return None
        return self.rhs[self.dot]

    @property
    def key(self) -> EarleyKey:
        return EarleyKey(self.lhs, self.rhs, self.dot, self.start, self.end)

    def advance(self, child: ParseNode, add_log_prob: float, new_end: int) -> "EarleyState":
        return EarleyState(
            lhs=self.lhs,
            rhs=self.rhs,
            dot=self.dot + 1,
            start=self.start,
            end=new_end,
            log_prob=self.log_prob + add_log_prob,
            children=self.children + [child],
        )

    def to_tree(self) -> ParseNode:
        return ParseNode(
            label=self.lhs,
            prob=self.log_prob,
            children=self.children
        )

class ProbabilisticEarleyParser:

    def __init__(self, grammar):
        self.grammar = grammar
        self.nonlex_by_lhs: dict[str, list] = defaultdict(list)
        self.lexical_rules: list = []

        for rule in grammar.rules:
            if rule.is_lexical:
                self.lexical_rules.append(rule)
            else:
                self.nonlex_by_lhs[rule.lhs].append(rule)

        self.nonterminals = set(self.nonlex_by_lhs.keys())
        self.nonterminals.update(rule.lhs for rule in self.lexical_rules)


    def preprocess(self, question: str) -> list[str]:
        text = self.grammar.normalize(question)

        replacements = {
            "da-mi": "da",
            "da mi": "da",
            "arata-mi": "arata",
            "arata mi": "arata",
            "afiseaza-mi": "afiseaza",
            "prezinta-mi": "prezinta",
            "care sunt": "care_sunt",
            "care este": "care_este",
            "ce este": "ce_este",

            "mai mare decat": "mai_mare_decat",
            "mai mare de": "mai_mare_decat",
            "mai mult decat": "mai_mare_decat",
            "mai mult de": "mai_mare_decat",
            "mai batrani de": "mai_mare_decat",
            "mai batran de": "mai_mare_decat",
            "in varsta de peste": "peste",
            "mai mic decat": "mai_mic_decat",
            "mai mic de": "mai_mic_decat",
            "mai mica de": "mai_mic_decat",
            "mai mici de": "mai_mic_decat",
            "mai putin decat": "mai_mic_decat",
            "mai putin de": "mai_mic_decat",
            "mai tineri de": "mai_mic_decat",
            "mai tanar de": "mai_mic_decat",
            "mai scurte de": "mai_mic_decat",
            "mai scump de": "mai_mare_decat",
            "mai scumpe de": "mai_mare_decat",
            "mai lungi de": "mai_mare_decat",
            "egal cu": "egal_cu",

            "alba iulia": "alba_iulia",
            "baia mare": "baia_mare",
            "satu mare": "satu_mare",
            "sfantu gheorghe": "sfantu_gheorghe",
            "targu mures": "targu_mures",
            "piatra neamt": "piatra_neamt",
            "ramnicu valcea": "ramnicu_valcea",
            "drobeta-turnu severin": "drobeta_turnu_severin",
            "drobeta turnu severin": "drobeta_turnu_severin",
        }

        for src, dst in replacements.items():
            text = text.replace(src, dst)

        return re.findall(r"[a-z0-9_=-]+", text)

    def match_terminal(self, symbol: str, token: str) -> bool:
        return symbol == token

    def _push_state(self, chart_col: dict[EarleyKey, EarleyState], state: EarleyState) -> bool:
        old = chart_col.get(state.key)
        if old is None or state.log_prob > old.log_prob:
            chart_col[state.key] = state
            return True
        return False

    def _lexical_matches_for_nonterminal(self, nonterminal: str, token: str) -> list:
        matches = []

        lex_rules = self.grammar.get_lex_rules_for(token)
        for rule in lex_rules:
            if rule.lhs == nonterminal:
                matches.append(rule)

        return matches


    def parse(self, question: str, start_symbol: str = "S") -> ParseResult:
        tokens = self.preprocess(question)
        n = len(tokens)

        if n == 0:
            return ParseResult(
                question=question,
                tokens=[],
                success=False,
                tree=None,
                probability=0.0,
                error="Întrebare goală",
            )

        chart: list[dict[EarleyKey, EarleyState]] = [dict() for _ in range(n + 1)]
        agenda: list[deque[EarleyState]] = [deque() for _ in range(n + 1)]

        gamma_rule_rhs = (start_symbol,)
        start_state = EarleyState(
            lhs="Γ",
            rhs=gamma_rule_rhs,
            dot=0,
            start=0,
            end=0,
            log_prob=0.0,
            children=[],
        )
        self._push_state(chart[0], start_state)
        agenda[0].append(start_state)

        for i in range(n + 1):
            while agenda[i]:
                state = agenda[i].popleft()

                if state.is_complete:
                    completed_tree = state.to_tree()

                    for waiting in list(chart[state.start].values()):
                        next_sym = waiting.next_symbol
                        if next_sym == state.lhs:
                            advanced = waiting.advance(
                                child=completed_tree,
                                add_log_prob=state.log_prob,
                                new_end=i,
                            )
                            if self._push_state(chart[i], advanced):
                                agenda[i].append(advanced)
                    continue

                next_sym = state.next_symbol
                assert next_sym is not None

                if next_sym in self.nonterminals:
                    for rule in self.nonlex_by_lhs.get(next_sym, []):
                        if rule.prob <= 0:
                            continue
                        predicted = EarleyState(
                            lhs=rule.lhs,
                            rhs=tuple(rule.rhs),
                            dot=0,
                            start=i,
                            end=i,
                            log_prob=math.log(rule.prob),
                            children=[],
                        )
                        if self._push_state(chart[i], predicted):
                            agenda[i].append(predicted)

                    if i < n:
                        token = tokens[i]
                        lex_rules = self._lexical_matches_for_nonterminal(next_sym, token)

                        for rule in lex_rules:
                            if rule.prob <= 0:
                                continue

                            leaf = ParseNode(
                                label=rule.lhs,
                                prob=math.log(rule.prob),
                                word=token,
                                children=[],
                            )

                            advanced = state.advance(
                                child=leaf,
                                add_log_prob=math.log(rule.prob),
                                new_end=i + 1,
                            )

                            if self._push_state(chart[i + 1], advanced):
                                agenda[i + 1].append(advanced)

                    continue

                if i < n and self.match_terminal(next_sym, tokens[i]):
                    leaf = ParseNode(
                        label=next_sym,
                        prob=0.0,
                        word=tokens[i],
                        children=[],
                    )
                    advanced = state.advance(
                        child=leaf,
                        add_log_prob=0.0,
                        new_end=i + 1,
                    )
                    if self._push_state(chart[i + 1], advanced):
                        agenda[i + 1].append(advanced)

        final_candidates = []
        for st in chart[n].values():
            if (
                st.lhs == "Γ"
                and st.is_complete
                and st.start == 0
                and st.end == n
            ):
                final_candidates.append(st)

        if not final_candidates:
            best = None
            for col in chart:
                for st in col.values():
                    if best is None or st.log_prob > best.log_prob:
                        best = st

            return ParseResult(
                question=question,
                tokens=tokens,
                success=False,
                tree=best.to_tree() if best else None,
                probability=math.exp(best.log_prob) if best else 0.0,
                error="Nu s-a găsit o parsare completă pentru simbolul de start S",
            )

        best = max(final_candidates, key=lambda s: s.log_prob)

        tree = best.children[0] if best.children else None

        return ParseResult(
            question=question,
            tokens=tokens,
            success=True,
            tree=tree,
            probability=math.exp(best.log_prob),
            error="",
        )