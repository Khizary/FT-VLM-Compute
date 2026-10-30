"""Answer normalisation, grouping and grading for Task 3 (fixed before final evaluation).

MATH: math-verify equivalence (pinned 0.9.0). Answers are grouped without gold: a new answer joins the first
existing group whose representative it is equivalent to (verify in either direction); grading uses the same rule
against the gold answer.
MathVista: the official normalize_extracted_answer rules (question type, answer type, precision, choices) ported
from https://github.com/lupantech/MathVista/blob/main/evaluation/calculate_score.py; multiple-choice answers are
represented as option letters. Exact match of normalised strings.
Missing extractions -> one NO_ANSWER group (always incorrect). For MathVista, non-missing extractions whose
normalisation fails (e.g. "5-10 years" for an integer question) -> one INVALID group (always incorrect).
"""
import re

NO_ANSWER = "NO_ANSWER"
INVALID = "INVALID"
SCORER = {"math": "math-verify==0.9.0 (parse with $...$ wrapping, verify either direction for grouping, verify(gold, pred) for grading)",
          "mathvista": "MathVista calculate_score.normalize_extracted_answer (main branch, Oct 2026) + option letters + exact match"}


# ---------------- MATH ----------------
_cache = {}


def mv_parse(s):
    from math_verify import parse

    if s not in _cache:
        try:
            _cache[s] = parse("$" + s + "$", parsing_timeout=5)
        except Exception:
            _cache[s] = []
    return _cache[s]


def mv_equiv(a, b):
    from math_verify import verify

    pa, pb = mv_parse(a), mv_parse(b)
    if not pa or not pb:
        return " ".join(a.split()) == " ".join(b.split())
    try:
        return bool(verify(pa, pb, timeout_seconds=5) or verify(pb, pa, timeout_seconds=5))
    except Exception:
        return False


def mv_correct(gold, pred):
    from math_verify import verify

    pg, pp = mv_parse(gold), mv_parse(pred)
    if not pg or not pp:
        return " ".join(gold.split()) == " ".join(pred.split())
    try:
        return bool(verify(pg, pp, timeout_seconds=5))
    except Exception:
        return False


def group_math(answers, gold):
    """answers: list (rollout order) of str|None -> (group label per rollout, {label: correct})"""
    reps, labels = [], []
    for a in answers:
        if a is None or not str(a).strip():
            labels.append(NO_ANSWER)
            continue
        for k, r in enumerate(reps):
            if a == r or mv_equiv(r, a):
                labels.append(f"g{k}")
                break
        else:
            reps.append(a)
            labels.append(f"g{len(reps) - 1}")
    correct = {f"g{k}": mv_correct(gold, r) for k, r in enumerate(reps)}
    correct[NO_ANSWER] = False
    reps_map = {f"g{k}": r for k, r in enumerate(reps)}
    return labels, correct, reps_map


# ---------------- MathVista ----------------
def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def get_most_similar(prediction, choices):
    d = [levenshtein(prediction, c) for c in choices]
    return choices[d.index(min(d))]


def normalize_mathvista(extraction, meta):
    """official rules; returns the normalised string (option letter for multi-choice) or INVALID"""
    choices, qtype, atype, prec = meta.get("choices"), meta["question_type"], meta["answer_type"], meta.get("precision")
    if qtype == "multi_choice":
        extraction = extraction.strip() if isinstance(extraction, str) else str(extraction)
        letter = re.findall(r"\(([a-zA-Z])\)", extraction)
        if letter:
            extraction = letter[0].upper()
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        text = choices[letters.index(extraction)] if extraction in letters else get_most_similar(extraction, choices)
        return letters[choices.index(text)]
    try:
        if atype == "integer":
            return str(int(float(extraction)))
        if atype == "float":
            return str(round(float(extraction), int(prec)))
        if atype == "list":
            return str(extraction)
    except Exception:
        return INVALID
    return str(extraction)


def gold_mathvista(meta):
    if meta["question_type"] == "multi_choice":
        return chr(ord("A") + meta["choices"].index(meta["answer"]))
    return str(meta["answer"])


def group_mathvista(answers, meta):
    gold = gold_mathvista(meta)
    labels = [NO_ANSWER if (a is None or not str(a).strip()) else normalize_mathvista(a, meta) for a in answers]
    correct = {l: (l == gold and l not in (NO_ANSWER, INVALID)) for l in set(labels)}
    return labels, correct, {l: l for l in set(labels)}
