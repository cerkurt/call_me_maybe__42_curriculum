from llm_sdk import Small_LLM_Model
from . import parameter_extractors, llm_helpers

import re
import string


def regex_candidate_generator(
    user_request: str, source_string: str
) -> dict[str, str]:
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
    open_assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    candidates = regex_candidate_generator(user_request, source_string)

    pattern_label_codes: dict = {}
    for index, label in enumerate(candidates):
        pattern_label_codes[chr(65 + index)] = label

    encoded_labels: dict = {}
    for label, title in pattern_label_codes.items():
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

    selected_label = pattern_label_codes[selected_candidate]
    selected_pattern = candidates[selected_label]

    return selected_pattern


def convert_regex_replacement_value(
    user_request: str, replacement: str
) -> str:
    char_name_dict: dict = {
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

    for index in range((len(substrings) - 1)):
        current_piece = substrings[index]
        next_piece = substrings[index + 1]
        if current_piece.lower() == "with":
            symbol_name = next_piece.rstrip(".,!?;:").lower()
            if (
                symbol_name == replacement.lower()
                and symbol_name in char_name_dict
            ):
                return char_name_dict[symbol_name]

    return replacement
