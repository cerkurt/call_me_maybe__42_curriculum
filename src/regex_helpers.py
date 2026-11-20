"""Build and select a bounded regex catalog and interpret replacement names.

Combine supported character categories with eligible literals from the
request. Use constrained option-code selection to return a pattern, and
normalize a small set of unquoted replacement names separately.
These helpers do not execute the requested text replacement.
"""

from llm_sdk import Small_LLM_Model

from . import parameter_extractors, llm_helpers

import re
import string


def regex_candidate_generator(
    user_request: str, source_string: str
) -> dict[str, str]:
    """Map readable regex categories and eligible literals to patterns.

    Args:
        user_request: Request from which to collect completed quoted spans.
        source_string: Text in which a requested pattern should match.

    Returns:
        A dictionary containing patterns for vowels, letters, digit sequences,
        uppercase letters, lowercase letters, whitespace, and punctuation.
        Add each quoted literal that occurs in source_string but is not the
        whole source, escaping regex metacharacters with re.escape().

    Labels describe patterns for model selection; they are not output regex
    values themselves. This finite catalog does not cover arbitrary regexes.
    """
    literals = parameter_extractors.string_splitter_inside_the_quotes(
        user_request
    )

    available_patterns = {
        "vowels": "[aeiouAEIOU]",
        "letters": "[a-zA-Z]+",
        "digit sequences": "[0-9]+",
        "uppercase letters": "[A-Z]",
        "lowercase letters": "[a-z]",
        "whitespace": r"\s+",
        "punctuation": "[" + re.escape(string.punctuation) + "]",
    }

    for string_literal in literals:
        if string_literal in source_string and string_literal != source_string:
            # Escape metacharacters: a literal 'a.b' must not match 'aXb'.
            available_patterns[f"literal: {string_literal}"] = re.escape(
                string_literal
            )

    return available_patterns


def generate_regex_pattern(
    model_object: Small_LLM_Model,
    user_request: str,
    source_string: str,
    replacement: str,
) -> str:
    """Choose a catalog pattern using short, constrained option codes.

    Args:
        model_object: Existing SDK model shared with the calling pipeline.
        user_request: Request describing the desired match/replacement.
        source_string: Source text shown to the model as selection context.
        replacement: Replacement value shown as additional context.

    Returns:
        The regex string associated with the selected option, not the
        option's letter or descriptive label.

    Assign successive letters starting at A to catalog labels, encode those
    letters, and ask generate_name_from_candidates() to choose one within
    10 tokens. Resolve the selected code to its label and then its pattern.
    Selection errors propagate; no replacement operation is executed here.
    """
    open_assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    candidates = regex_candidate_generator(user_request, source_string)

    pattern_label_codes: dict[str, str] = {}
    # Keep short answer codes separate from labels and their actual regexes.
    for index, label in enumerate(candidates):
        pattern_label_codes[chr(65 + index)] = label

    encoded_labels: dict[str, list[int]] = {}
    for label, title in pattern_label_codes.items():
        # Encode the option code, since that is what the model must return.
        encoded_labels[label] = model_object.encode(label)[0].tolist()

    readable_options = ""
    for code, description in pattern_label_codes.items():
        readable_options += f"{code}: {description}\n"

    selection_message = (
        llm_helpers.format_request_message(
            "system",
            "Choose what the user wants to find and replace in the source "
            "text. Return only the letter of the best option. "
            "Do not execute the replacement.",
        )
        + llm_helpers.format_request_message(
            "user",
            f"Request: {user_request}\n"
            f"Source text: {source_string}\n"
            f"Replacement value: {replacement}\n"
            "Options:\n"
            f"{readable_options}"
            "Which option describes what must be found? "
            "Answer with only its letter.",
        )
        + open_assistant_start
    )

    selected_candidate = llm_helpers.generate_name_from_candidates(
        model_object, selection_message, encoded_labels, 10
    )

    # Resolve both mappings: selected code -> readable label -> regex pattern.
    selected_label = pattern_label_codes[selected_candidate]
    selected_pattern = candidates[selected_label]

    return selected_pattern


def convert_regex_replacement_value(
    user_request: str, replacement: str
) -> str:
    """Convert supported unquoted character names following with to symbols.

    Args:
        user_request: Original request, split without losing quoted chunks.
        replacement: Extracted replacement string to interpret.

    Returns:
        The symbol for asterisk(s), hyphen(s), or underscore(s) when an
        unquoted name immediately follows with and matches replacement,
        ignoring case. Strip trailing .,!?:; punctuation from that request
        chunk for comparison. Otherwise return replacement unchanged.

    Quoted names such as 'asterisks' remain literal text. This is a small
    English with-name rule, not a general natural-language symbol converter.
    """
    char_name_dict: dict[str, str] = {
        "asterisk": "*",
        "asterisks": "*",
        "hyphen": "-",
        "hyphens": "-",
        "underscore": "_",
        "underscores": "_",
    }

    substrings = parameter_extractors.string_splitter_out_of_quotes(
        user_request
    )

    # Stop one position early because each iteration also reads index + 1.
    for index in range((len(substrings) - 1)):
        current_piece = substrings[index]
        next_piece = substrings[index + 1]
        if current_piece.lower() == "with":
            # Quotes remain in the chunk, so 'asterisks' stays literal text.
            symbol_name = next_piece.rstrip(".,!?;:").lower()
            if (
                symbol_name == replacement.lower()
                and symbol_name in char_name_dict
            ):
                return char_name_dict[symbol_name]

    # Preserve extracted values that do not match the supported symbol rule.
    return replacement
