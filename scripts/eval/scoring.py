"""The answer scorer used for every method and every number in the paper (Appendix B.3).

`score_item(pred, gold, question, answer_type, n_options)` returns `(base, audited)`:

  base     the shared soft-match rules of Dynamic Cheatsheet (Suzgun et al.), vendored
           below under their MIT license, applied identically to every method
  audited  base OR the recoveries below, also applied identically to every method;
           this is the column reported in the paper
             integer (AIME)     exact match after removing punctuation and whitespace
             math (MATH)        + normalized / numerically equal LaTeX, element-wise for
                                  lists, tuples and matrices, relative tolerance 1e-6
             mcq_letter         + the option's full text, or its numbers in order, when
                                  it identifies a unique option
             free (HLE-Exact    canonical exact match
              numbers)

An item with no parsable answer is incorrect, and the denominator is the full stream.
"""
import re

from lere.answers import math_500_equiv, normalize_mcq, option_texts


# ---------------------------------------------------------------------------------------
# Dynamic Cheatsheet evaluation rules (dynamic_cheatsheet/utils/evaluation.py).
# MIT License, Copyright (c) 2025 Mirac Suzgun. Reproduced unchanged.
# ---------------------------------------------------------------------------------------
def remove_punctuation(output: str) -> str:
    """
    Remove punctuation from the output.
    """
    markers = [",", ";", ":", ".", '"']
    for marker in markers:
        output = output.replace(marker, "")
    return output


def convert_newline_to_space(output: str) -> str:
    """
    Convert newline to space.
    """
    output = output.replace("\n", " ")
    return output


def eval_for_exact_matching_with_no_punctuation(
    output: str, target: str
) -> bool:
    """
    Evaluate if the output is exactly the same as the target.
    """
    output = remove_punctuation(output)
    output = convert_newline_to_space(output)
    if target == output:
        return True
    return False


def eval_for_math_500(output: str, target: str) -> bool:
    """
    Evaluate MATH_500 with layered normalization:
    1. Strip all whitespace (handles LaTeX spacing variants)
    2. Strip \\text{} wrappers (target may have \\text{Evelyn}, model says Evelyn)
    3. Strip degree/percent unit suffixes (90^\\circ vs 90, 10\\% vs 10)
    4. Numeric float normalization (8 vs 8.0, .35 vs 0.35)
    """
    def normalize(s: str) -> str:
        s = re.sub(r'\s+', '', s.strip())
        return s

    def strip_text_cmd(s: str) -> str:
        return re.sub(r'\\text\{([^}]*)\}', lambda m: m.group(1), s)

    def strip_units(s: str) -> str:
        # Remove ^\circ, \circ, \degree, \%, %
        return re.sub(r'\^\{?\\circ\}?|\\circ|\\degree|\\%', '', s)

    def try_numeric(a: str, b: str) -> bool:
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (ValueError, TypeError):
            return False

    n_out = normalize(output)
    n_tgt = normalize(target)

    # Pass 1: whitespace only
    if n_out == n_tgt:
        return True

    # Pass 2: strip \text{} wrappers
    if normalize(strip_text_cmd(n_out)) == normalize(strip_text_cmd(n_tgt)):
        return True

    # Pass 3: strip unit suffixes
    if normalize(strip_units(n_out)) == normalize(strip_units(n_tgt)):
        return True

    # Pass 4: numeric comparison
    if try_numeric(n_out, n_tgt):
        return True

    return False


def eval_for_multiple_choice(input_text: str, final_answer: str, target: str) -> bool:
    """
    Evaluates if the final answer matches the target using pattern matching.
    
    Args:
        input_text (str): The original question text including options
        final_answer (str): The model's answer
        target (str): The correct answer
    
    Returns:
        bool: True if answer is correct, False otherwise
    """
    # Handle empty or None inputs
    if not final_answer or not target:
        return False
    
    def clean_text(text: str) -> str:
        if not text:
            return ""
        return text.lower().strip().replace('`', '').replace('(', '').replace(')', '')
    
    def extract_option_text(input_text: str, option_letter: str) -> str:
        try:
            # Try different formats of options sections
            options_section = ""
            if 'options:' in input_text.lower():
                options_section = input_text.lower().split('options:')[1].strip()
            elif 'choices:' in input_text.lower():
                options_section = input_text.lower().split('choices:')[1].strip()
            
            if not options_section:
                # Try to find options in the format (A) text, (B) text
                lines = input_text.lower().split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith(f'({option_letter})') or line.strip().startswith(f'{option_letter})'):
                        return line.split(')', 1)[1].strip()
                
            # Process the options section if found
            for line in options_section.split('\n'):
                line = line.strip()
                if line.startswith(f'({option_letter})') or line.startswith(f'{option_letter})'):
                    return line.split(')', 1)[1].strip()
                # Handle options like "A. text" format
                if line.startswith(f'{option_letter}.'):
                    return line.split('.', 1)[1].strip()
        except:
            return ''
        return ''

    # Full option match (A), (B), etc. (e.g., (A) == (A))
    if final_answer == target:
        return True

    # Clean and normalize inputs
    clean_answer = clean_text(final_answer)
    clean_target = clean_text(target)
    
    # Handle target formats: (A), A), A, etc.
    target_letter = ""
    if len(clean_target) == 1:
        target_letter = clean_target
    elif clean_target.endswith(')'):
        target_letter = clean_target[-2]
    else:
        # Extract the last character if it's a letter a-d or A-D
        last_char = clean_target[-1]
        if last_char in 'abcd':
            target_letter = last_char
    
    # Direct letter match (a, b, c, d)
    if len(clean_answer) == 1 and clean_answer in 'abcd' and clean_answer == target_letter:
        return True
    
    # Handle answer formats like "A" or "A."
    if clean_answer.startswith(target_letter) and (len(clean_answer) == 1 or 
                                                  (len(clean_answer) == 2 and clean_answer[1] == '.')):
        return True
    
    # Handle answer formats like "Option A" or "Answer is A"
    if clean_answer.endswith(target_letter) and (clean_answer[-2:] == f" {target_letter}" or 
                                               clean_answer[-3:] == f" {target_letter}."):
        return True
    
    # Text content match - check if the target option text is in the answer
    target_text = extract_option_text(input_text, target_letter)
    
    if target_text and target_text in clean_answer:
        return True
    
    # Handle numerical answers (if target is a number and answer contains that number)
    if target_letter.isdigit() and target_letter in clean_answer:
        return True
        
    return False


_FREE_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_FREE_PREFIX_RE = re.compile(r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s*", re.IGNORECASE)


def canonical_free_answer(text: str) -> str:
    s = str(text).strip()
    m = _FREE_BOXED_RE.search(s)
    if m:
        s = m.group(1)
    s = s.replace("$", "").replace("\\!", "").replace("\\,", "")
    s = s.strip().strip("`").strip()
    s = _FREE_PREFIX_RE.sub("", s)
    s = s.strip().rstrip(".").strip()
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()


def eval_for_free_answer(output: str, target: str) -> bool:
    if output is None or target is None:
        return False
    return canonical_free_answer(output) == canonical_free_answer(target)

# ---------------------------------------------------------------------------------------
# MATH: stronger LaTeX equivalence (notation-insensitive), used for the audited column.
# ---------------------------------------------------------------------------------------
CC = (("\x0c", "\\f"), ("\t", "\\t"), ("\x08", "\\b"), ("\r", "\\r"))

def norm(s):
    s = str(s)
    for cc, cmd in CC: s = s.replace(cc, cmd)                                   # JSON escape damage
    s = s.replace("\\\\", "\\")                                                 # over-escaped \\\\sqrt (ACE JSON)
    s = re.sub(r"\\text\{\s*\(?([A-Za-z])\)?\s*\}", r"\1", s)                    # \text{(C)} -> C  (before unit strip)
    s = re.sub(r"^\s*\\(?:mbox|text)\{([^}]*)\}\s*$", r"\1", s)                    # \text{Circle} IS the answer, not a unit
    s = re.sub(r"\\(?:mbox|text)\{[^}]*\}(\^\{?\d\}?)?", "", s)                  # units: \mbox{ inches}^2
    s = s.replace("\\!", "").replace("\\,", "").replace("\;", "").replace("\\$", "").replace("$", "")
    s = re.sub(r"\^\{?\\circ\}?|\\circ|\\degree|\\%", "", s)
    s = re.sub(r"^\s*(?:[a-zA-Z]|\\[a-zA-Z]+)\s*(?:=|\\in)\s*", "", s)           # x=5, \lambda \in ..
    s = re.sub(r"\\d?frac\s*(\d)\s*(\d)(?!\d)", r"\\frac{\1}{\2}", s)            # \frac 59 -> \frac{5}{9}
    s = s.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"_\{(\w+)\}", r"_\1", s)
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(\d),(?=\d{3}(?!\d))", r"\1", s)                                # 10,080
    # inequality chain  a<x<=b  ->  (a,b]
    m = re.fullmatch(r"(-?[\d.\\a-z{}]+)(<|\\le|\\leq)[\\a-zA-Z]+(<|\\le|\\leq)(-?[\d.\\a-z{}]+)", s)
    if m: s = ("[" if m.group(2) != "<" else "(") + m.group(1) + "," + m.group(4) + ("]" if m.group(3) != "<" else ")")
    # Outer brackets. A SET \{a,b\} is order-free, so its braces are dropped and the members
    # sorted below. An INTERVAL / tuple keeps its bracket TYPES -- (3,4] is not [3,4] -- so
    # they are kept as an ordered prefix and the members are NOT sorted.
    ordered = False
    if len(s) > 1 and s[0] in "{([" and s[-1] in "})]" and "," in s:
        if s[0] == "{" and s[-1] == "}": s = s[1:-1]
        else: ordered = True; s, tag = s[1:-1], s[0] + s[-1]
    # expand a ± PER comma-part (a set may mix ± terms with plain ones), balancing braces
    out = []
    for part in s.split(","):
        m = re.fullmatch(r"(.+?)\\pm(.+)", part)
        if m: out += [m.group(1) + "+" + m.group(2), m.group(1) + "-" + m.group(2)]
        else: out.append(part)
    def canon(p):                                                                # commutative sums
        def sort_sum(expr):
            if "+" in expr and "-" not in expr.replace("+-", ""): return "+".join(sorted(expr.split("+")))
            return expr
        # inside every \frac{numerator}{...}
        p = re.sub(r"\\frac\{([^{}]*)\}", lambda m: "\\frac{" + sort_sum(m.group(1)) + "}", p)
        return sort_sum(p) if "{" not in p else p
    parts = [canon(p) for p in out]
    if ordered: return tag + ":" + ",".join(parts)
    parts = sorted(parts)
    return ",".join(parts)

def latex_val(s):
    """Numeric value of a simple exact LaTeX/Python expression, or None. Handles \\frac,
    \\sqrt, \\pi, implicit products (3\\sqrt{5}, 12\\pi), sums, a/b, decimals. Used
    on BOTH sides so a decimal prediction can match an exact gold (or vice versa)."""
    import sympy
    t = str(s).strip()
    if not t or len(t) > 80 or "=" in t or "," in t or "\\begin" in t: return None
    t = re.sub(r"\^\{?\\circ\}?|\\circ|\\degree|\\%", "", t)
    t = t.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "").replace("\\$", "")
    t = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", t)
    t = re.sub(r"\\sqrt\s*(\d+)", r"sqrt(\1)", t)
    for _ in range(3):                                                        # nested fracs
        t = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\d?frac\s*(\d)\s*(\d)", r"((\1)/(\2))", t)
    t = t.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    t = re.sub(r"[{}]", "", t)
    if re.search(r"[\\A-Za-hj-oq-z]", t.replace("sqrt", "").replace("pi", "")): return None   # leftover symbols/variables
    t = re.sub(r"(\d)\s*(sqrt|pi|\()", r"\1*\2", t); t = re.sub(r"\)\s*(sqrt|pi|\(|\d)", r")*\1", t); t = re.sub(r"(pi|\))\s*sqrt", r"\1*sqrt", t)
    t = re.sub(r"\s+", "", t)
    try:
        v = sympy.sympify(t, locals={"pi": sympy.pi, "sqrt": sympy.sqrt, "i": sympy.I})
        return complex(sympy.N(v, 15)) if v.free_symbols == set() else None
    except Exception:
        return None

def pmatrix_flat(s):
    t = str(s)
    if "pmatrix" in t:
        body = re.sub(r"\\(begin|end)\{pmatrix\}", "", t)
        return [x.strip() for x in re.split(r"\\\\|&", body) if x.strip()]
    if re.fullmatch(r"\s*\[\s*\[.*\]\s*\]\s*|\s*\[.*\]\s*", t) and "," in t:
        return [x.strip() for x in re.split(r"[\[\],]", t) if x.strip()]
    return None

def strong_equiv(p, g):
    if math_500_equiv(p, g): return True
    a, b = norm(p), norm(g)
    if a == b: return True
    va, vb = latex_val(p), latex_val(g)
    if va is not None and vb is not None:
        return abs(va - vb) <= 1e-6 * max(1.0, abs(vb))
    # element-wise numeric comparison of two sets/lists (e.g. decimal roots vs 1 ± \\sqrt{19})
    if "," in a and "," in b:
        xa, xb = a.split(":")[-1].split(","), b.split(":")[-1].split(",")
        if len(xa) == len(xb) and (a.split(":")[0] == b.split(":")[0] if ":" in a or ":" in b else True):
            va, vb = [latex_val(x) for x in xa], [latex_val(x) for x in xb]
            if all(v is not None for v in va + vb):
                key = lambda z: (z.real, z.imag)
                ordered = ":" in a or ":" in b                                   # tuple/interval: keep order
                pa, pb = (va, vb) if ordered else (sorted(va, key=key), sorted(vb, key=key))
                if all(abs(x - y) <= 1e-6 * max(1.0, abs(y)) for x, y in zip(pa, pb)): return True
    fa, fb = pmatrix_flat(p), pmatrix_flat(g)
    if fa and fb and len(fa) == len(fb):
        vals = [(latex_val(x), latex_val(y)) for x, y in zip(fa, fb)]
        return all(x is not None and y is not None and abs(x - y) <= 1e-6 * max(1.0, abs(y)) for x, y in vals)
    return False

# ---------------------------------------------------------------------------------------
# Multiple choice: option-text and numeric recovery, used for the audited column.
# ---------------------------------------------------------------------------------------
SCI   = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:[×x*]\s*10\s*\^?\s*|[eE])\s*([-+]?\d+)")
PLAIN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

def nums(s):
    s, out, spans = (s or ""), [], []
    for m in SCI.finditer(s):
        out.append(float(m.group(1)) * 10 ** int(m.group(2))); spans.append(m.span())
    for m in PLAIN.finditer(s):
        if any(a <= m.start() < b for a, b in spans):
            continue
        try: out.append(float(m.group(0).replace(",", "")))
        except ValueError: pass
    return out

def eq(a, b, rel=1e-3):
    if a == b: return True
    d = max(abs(a), abs(b))
    return d > 0 and abs(a - b) / d <= rel

def clean(s):
    return (s or "").strip().lower().replace("`", "").replace("(", "").replace(")", "")

def paren_match(pred, gold_letter, question):
    t = option_texts(question).get((gold_letter or "").upper()) or ""
    return bool(t.strip()) and clean(t) in clean(pred)

def numeric_match(pred, gold_letter, question):
    opts = option_texts(question)
    g = opts.get((gold_letter or "").upper())
    if not g: return False
    pn, gn = nums(pred), nums(g)
    if not (pn and gn and eq(pn[0], gn[0])): return False
    for L, t in opts.items():                       # unique-option requirement
        if L == gold_letter.upper(): continue
        tn = nums(t)
        if tn and eq(tn[0], pn[0]): return False
    return True

# runs/*/items.jsonl carries only id/gold/question -- answer_type and n_options live in

# ---------------------------------------------------------------------------------------
# The shared scorer.
# ---------------------------------------------------------------------------------------
def score_item(pred, gold, question, answer_type, n_options):
    """(base, audited) under the shared standard."""
    pred = "" if pred is None else str(pred)
    gold = str(gold)
    if answer_type == "math":
        base = bool(eval_for_math_500(pred, gold))
        return base, base or strong_equiv(pred, gold)
    if answer_type == "integer":
        base = bool(eval_for_exact_matching_with_no_punctuation(pred.lower(), gold.lower()))
        return base, base
    if answer_type == "free":               # HLE-Exact numbers: canonical exact match
        base = bool(eval_for_free_answer(pred, gold))
        return base, base
    # mcq_letter: DC scorer as run (identical to LeRe's is_correct on letters + option text)
    base = bool(eval_for_multiple_choice(question, pred, gold))
    if base or not pred.strip():
        return base, base
    g = normalize_mcq(gold, n_options)
    if not g or normalize_mcq(pred, n_options) is not None:
        return base, base
    rec = paren_unique_match(pred, g, question) or numeric_seq_match(pred, g, question)
    return base, bool(rec)


def _nums(s):
    """nums with the sign attached when it is written as '- 1.73' (space after minus)."""
    return nums(re.sub(r"(?<![\w)])-\s+(?=\d)", "-", str(s)))


_UNIT_NOISE = re.compile(r"\\(?:times|cdot|text|mathrm|mbox|left|right|frac|sqrt|circ|degree|,|;|!)|[\d.,eE+\-*/^(){}\[\]$%°~\s×·]|\bx\b")


def _unit_residue(s):
    """Letters that remain once numbers, LaTeX commands and punctuation are gone: the unit."""
    return re.sub(r"[^a-z]", "", _UNIT_NOISE.sub(" ", str(s).lower()))


def paren_unique_match(pred, gold_letter, question):
    """audit_mcq_scoring.paren_match plus a uniqueness requirement: no OTHER option's
    text may also be contained in the prediction (an option that is a prefix of another
    would otherwise be credited when the prediction names the longer one)."""
    if not paren_match(pred, gold_letter, question):
        return False
    opts = option_texts(question)
    cp = clean(pred)
    contained = {L: len(clean(t)) for L, t in opts.items() if t.strip() and clean(t) in cp}
    g = gold_letter.upper()
    # the gold option must be the single longest option the prediction contains
    return g in contained and all(n < contained[g] for L, n in contained.items() if L != g)


def numeric_seq_match(pred, gold_letter, question):
    """Stricter than audit_mcq_scoring.numeric_match: EVERY number in the gold option must
    appear in the prediction, in order, at 0.1% tolerance (not just the leading one), so
    "0.857 and 0.857" does not get credit for option "0.857 and 0.926". Unique-option rule
    kept: no other option may share the same numeric sequence."""
    opts = option_texts(question)
    g = opts.get((gold_letter or "").upper())
    if not g:
        return False
    pn, gn = _nums(pred), _nums(g)
    if not pn or not gn or len(pn) < len(gn):
        return False
    if not all(eq(a, b) for a, b in zip(pn[:len(gn)], gn)):
        return False
    if len(pn) != len(gn):
        return False
    # unit check: if both sides name a unit, it must be the same unit ("4.8 km" is not "4.8 m")
    up, ug = _unit_residue(pred), _unit_residue(g)
    if up and ug and up != ug:
        return False
    for L, t in opts.items():
        if L == gold_letter.upper():
            continue
        tn = _nums(t)
        if tn and len(tn) == len(gn) and all(eq(a, b) for a, b in zip(tn, gn)):
            return False
    return True
