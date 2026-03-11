"""
T5 tokenizer loading and concept-to-token mapping.

Handles initialisation of the T5 tokenizer and provides functions
for mapping concepts to their token representations, as well as
building the sorted token list used for the tokenization tab.
"""


def load_tokenizer():
    """
    Load the T5-v1.1-XXL tokenizer.

    Returns the tokenizer instance, or ``None`` if loading fails.
    """
    try:
        from transformers import T5Tokenizer
        import logging

        logging.getLogger("transformers").setLevel(logging.ERROR)
        return T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    except Exception as e:
        print(f"Warning: Could not load tokenizer for analysis: {e}")
        return None


def compute_concept_tokens(tokenizer, concepts):
    """
    Map each concept to its T5 token list.

    Parameters
    ----------
    tokenizer : T5Tokenizer or None
    concepts : set[str]

    Returns
    -------
    dict[str, list[str]]
        Mapping from concept string to list of token strings.
    """
    if tokenizer is None:
        return {}
    concept_to_tokens = {}
    for c in concepts:
        tokens = tokenizer.tokenize(c)
        concept_to_tokens[c] = tokens
    return concept_to_tokens


def filter_real_tokens(tokens):
    """
    Filter out special characters (``▁`` / ``_``) that should not be
    counted as real tokens.
    """
    return [t for t in tokens if t not in ["\u2581", "_"]]


def build_concept_tokens_list(concept_to_tokens):
    """
    Build a sorted list of ``{concept, tokens, count}`` dicts,
    ordered by descending real-token count.
    """
    result = []
    for c, tokens in concept_to_tokens.items():
        real_tokens = filter_real_tokens(tokens)
        result.append({"concept": c, "tokens": tokens, "count": len(real_tokens)})
    result.sort(key=lambda x: x["count"], reverse=True)
    return result
