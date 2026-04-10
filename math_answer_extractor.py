from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from tokenize import TokenError
import re

RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
LATEX_FIXES = [
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]
RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")


# extracting the final answer box from the generated answer
def extract_final_answer_box(answer: str) -> str | None:
    box_start_idx = answer.rfind(r"\boxed")
    if box_start_idx == -1:
        return None
    current_idx = box_start_idx + len("r\boxed")
    while current_idx < len(answer) and answer[current_idx].isspace():
        current_idx += 1
    if current_idx == len(answer) or answer[current_idx] != "{":
        return None

    current_idx += 1
    brace_depth = 1  #'we opened {'
    content_start_idx = current_idx

    while current_idx < len(answer) and brace_depth > 0:
        if answer[current_idx] == "{":
            brace_depth += 1
        elif answer[current_idx] == "}":
            brace_depth -= 1
        current_idx += 1
    if brace_depth != 0:
        return None
    return answer[content_start_idx : current_idx - 1]


def extract_final_answer(answer, fallback="number_then_full"):
    box = extract_final_answer_box(answer)
    result = box or ""
    if result:
        return result
    if fallback:
        numbers = RE_NUMBER.findall(answer)
        if numbers:
            result = numbers[-1]
        elif fallback == "number_then_full":
            result = answer
    return result


def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("°", "")
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )

    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )
    return text.replace("{", "").replace("}", "").strip().lower()


# parser number: from string number to number
def sympy_parser(expr):
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,
                spp.implicit_multiplication_application,
            ),
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, IndexError, TokenError):
        return None


def split_into_parts(exp):
    if exp[0] in "[(" and exp[-1] in ")]" and "," in exp[1:-1]:
        exp = exp[1:-1].split(",")
        return [x.strip() for x in exp]
    return [exp] if exp else []


def answer_verifier(prediction, truth):
    if prediction == truth:
        return True
    prediction, truth = sympy_parser(prediction), sympy_parser(truth)
    if prediction and truth:
        try:
            if prediction - truth == 0:
                return True
        except Exception:
            pass
    return False
