"""Structural citation checks shared by generation and verification."""
import re


def normalize_citations(answer):
    return re.sub(
        r"\[([0-9]+(?:\s*,\s*[0-9]+)+)\]",
        lambda match: " ".join(f"[{number.strip()}]" for number in match[1].split(",")),
        answer,
    )


def citation_ids(answer):
    return list(dict.fromkeys(re.findall(r"\[[0-9]+\]", answer)))


def parse_passages(context):
    # Production headers are explicit. The fallback also supports plain tagged
    # passages used by callers of LLMClient without RetrievalEngine.
    pattern = r"(?m)^(\[\d+\]) Source:[^\n]*\n"
    matches = list(re.finditer(pattern, context))
    if not matches:
        matches = list(re.finditer(r"(?m)^(\[\d+\])[ \t]*", context))
    passages = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        label = match[1]
        if label in passages:
            return {}  # Ambiguous source identifiers cannot be verified.
        passages[label] = context[match.end():end].strip()
    return passages


def evidence_options(passages):
    """Give original paragraphs stable IDs so the model never retypes quotes."""
    options = {}
    for label, passage in passages.items():
        for index, paragraph in enumerate(re.split(r"\n\s*\n", passage), 1):
            quote = paragraph.strip()
            if quote:
                options[f"{label}:p{index}"] = {"citation_id": label, "quote": quote}
    return options


def resolve_evidence(verdict, options):
    if not isinstance(verdict, dict) or "evidence_ids" not in verdict:
        return verdict
    ids = verdict["evidence_ids"]
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(key, str) or key not in options for key in ids)):
        return {}
    evidence = [options[key] for key in dict.fromkeys(ids)]
    answer = verdict.get("answer")
    if not isinstance(answer, str):
        return {}
    # Citation-to-page mapping comes from source paragraph IDs, not model memory.
    answer = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\[[0-9]+\]", "", normalize_citations(answer)))
    answer = " ".join(answer.split())
    labels = list(dict.fromkeys(item["citation_id"] for item in evidence))
    punctuation = answer[-1] if answer[-1:] in (".", "!", "?") else ""
    body = answer[:-1].rstrip() if punctuation else answer
    return {**verdict, "answer": f"{body} {' '.join(labels)}{punctuation}", "evidence": evidence}


def verified_answer(draft, verdict, passages):
    """Accept citation repairs only, backed by literal quotes from cited sources.

    Quote membership is structural grounding; the verifier must still judge
    whether each quote actually entails the corresponding answer claim.
    """
    if not isinstance(verdict, dict) or verdict.get("supported") is not True:
        return None
    answer = verdict.get("answer")
    evidence = verdict.get("evidence")
    if not isinstance(answer, str) or not answer.strip() or not isinstance(evidence, list):
        return None
    answer = normalize_citations(answer)
    def strip_labels(text):
        text = " ".join(re.sub(r"\[[0-9]+\]", "", text).split())
        return re.sub(r"\s+([.,;:!?])", r"\1", text)
    if strip_labels(answer) != strip_labels(normalize_citations(draft)):
        return None
    labels = citation_ids(answer)
    if not labels:
        # Models sometimes return the correct answer and evidence map but omit
        # inline tags. Build tags from that map, then validate every quote below.
        mapped = [item.get("citation_id") for item in evidence if isinstance(item, dict)]
        if not mapped or any(not isinstance(label, str) for label in mapped):
            return None
        labels = list(dict.fromkeys(mapped))
        punctuation = answer[-1] if answer[-1:] in (".", "!", "?") else ""
        body = answer[:-1].rstrip() if punctuation else answer.rstrip()
        answer = f"{body} {' '.join(labels)}{punctuation}"
    if not labels or any(label not in passages for label in labels):
        return None
    grounded = set()
    for item in evidence:
        if not isinstance(item, dict):
            return None
        label, quote = item.get("citation_id"), item.get("quote")
        if not isinstance(label, str) or label not in labels or not isinstance(quote, str):
            return None
        quote = " ".join(quote.split())
        if not quote or quote not in " ".join(passages[label].split()):
            return None
        grounded.add(label)
    return answer.strip() if grounded == set(labels) else None
