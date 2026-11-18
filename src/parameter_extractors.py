"""Construct typed argument candidates and shared extraction prompts.

Collect quote-aware string fragments and word n-grams, parse supported
numeric forms into finite floats, and format the current argument request
with previously extracted values as context. Candidate building itself
does not invoke the LLM.
"""

from . import data_models, llm_helpers

import re
import json
import math


def string_splitter_out_of_quotes(text: str) -> list[str]:
    """Split text at spaces outside quoted chunks, retaining quote marks.

    Args:
        text: Natural-language request to divide into chunks.

    Returns:
        Nonempty chunks in input order. A single or double quote at the
        beginning of a chunk opens a quoted region; spaces inside that
        region remain in the same chunk. Outer quote marks are retained.

    Only literal spaces separate chunks here. This is a lightweight request
    splitter, not a general parser for escaped or nested quote delimiters.
    """
    completed_parts: list[str] = []
    current_part: list[str] = []
    is_quote_active = None

    for char in text:
        if is_quote_active is None:
            if char in ("'", '"') and not current_part:
                is_quote_active = char
        elif char == is_quote_active:
            is_quote_active = None

        if char == " " and is_quote_active is None:
            if current_part:
                completed_parts.append("".join(current_part))
                current_part = []
        else:
            current_part.append(char)

    if current_part:
        completed_parts.append("".join(current_part))

    return completed_parts


def string_splitter_inside_the_quotes(text: str) -> list[str]:
    """Collect completed single- or double-quoted spans without delimiters.

    Args:
        text: Natural-language request containing possible quoted values.

    Returns:
        Contents of completed quoted spans in their original order, including
        empty contents. Unclosed spans are not returned. Outside a span, an
        apostrophe immediately after a letter or digit does not open one.

    A matching quote closes the current span. Backslash escapes are not
    interpreted, so this is not a full parser for arbitrary quoted syntax.
    """
    completed_parts: list[str] = []
    current_part: list[str] = []
    is_quote_active = None

    for index, char in enumerate(text):
        if is_quote_active is None:
            if char in ("'", '"') and not (
                char == "'" and index > 0 and text[index - 1].isalnum()
            ):
                is_quote_active = char
        elif char == is_quote_active:
            completed_parts.append("".join(current_part))
            current_part = []
            is_quote_active = None
        else:
            current_part.append(char)

    return completed_parts


def string_slicer(user_request: str) -> list[str]:
    """Build a deduplicated pool of possible string argument values.

    Args:
        user_request: Original request from which to collect candidates.

    Returns:
        Candidate strings in first-seen order. Include split chunks with
        matching outer quotes removed, two- and three-word phrases outside
        recognized quotes, chunks with trailing .!? removed when changed,
        and completed quoted contents.

    Skip word n-grams identical to the whole request; do not append the full
    request as a separate fallback. Individual chunks can still equal it.
    Quoted empty or long values remain candidates, while arbitrary unquoted
    values longer than three words are not all enumerated. This function
    offers possible values without deciding which one a parameter needs.
    """
    inside_the_quotes_substrings = string_splitter_inside_the_quotes(
        user_request
    )
    out_of_the_quotes_substrings = string_splitter_out_of_quotes(user_request)

    word_filter_list: list[str] = []
    for word in out_of_the_quotes_substrings:
        if len(word) >= 2 and (
            (word[0] == "'" and word[len(word) - 1] == "'")
            or (word[0] == '"' and word[len(word) - 1] == '"')
        ):
            sliced_word = word[1 : (len(word) - 1)]
            word_filter_list.append(sliced_word)
        else:
            word_filter_list.append(word)

    segments = re.split(
        r"""(?<!\w)('[^']*'|"[^"]*")""",
        user_request,
    )

    for segment in segments[::2]:
        words = segment.split()

        for size in (2, 3):
            for start in range(len(words) - size + 1):
                candidate = " ".join(words[start : start + size])

                if candidate != user_request:
                    word_filter_list.append(candidate)

    for word in out_of_the_quotes_substrings:
        clean_word = word.rstrip(".!?")
        if clean_word != word and clean_word:
            word_filter_list.append(clean_word)

    word_filter_list.extend(inside_the_quotes_substrings)

    return list(dict.fromkeys(word_filter_list))


def extract_numbers_from_user_request(user_request: str) -> list[float]:
    """Extract supported numeric text as finite floating-point candidates.

    Args:
        user_request: Request containing possible numeric argument values.

    Returns:
        Floats in occurrence order, retaining duplicate values. Supported
        forms include signed integers, decimals with or without a leading
        zero, comma-grouped thousands, and scientific notation. Grouping
        commas are removed before conversion. No matches produce an empty
        list; number words are not converted.

    Raises:
        ValueError: Conversion produces a non-finite float.

    This helper extracts candidates, not their parameter assignments. Float
    conversion has the usual precision and range limits.
    """
    number_pattern = (
        r"(?<![\w.])"
        r"[-+]?"
        r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?"
        r"(?!\w)"
    )

    numbers_in_string = re.findall(number_pattern, user_request)

    float_list: list[float] = []

    for number in numbers_in_string:
        float_conversion = float(number.replace(",", ""))

        if math.isfinite(float_conversion):
            float_list.append(float_conversion)
        else:
            raise ValueError(
                "Non-finite number generated. "
                "Unsupported number to write in standard JSON."
            )

    return float_list


def message_for_parameter_extraction(
    user_request: str,
    function_object: data_models.FunctionDefinition,
    given_parameter_name: str,
    extracted_parameters: dict[str, object],
) -> str:
    """Format the shared extraction prompt and open JSON argument prefix.

    Args:
        user_request: Original natural-language request.
        function_object: Selected definition supplying metadata and types.
        given_parameter_name: Declared parameter to extract next.
        extracted_parameters: Previously extracted names and actual values.

    Returns:
        One string containing system/user messages and an open assistant
        response ending at the current JSON key's colon. There is no space
        after the colon; candidate construction supplies the leading space.

    Use a string-specific system instruction for preserving the complete
    input value, and the general instruction for other types. Add numeric
    sign guidance or boolean final-state guidance when applicable. Include
    all declarations and previous arguments as context, then serialize those
    arguments to start the next JSON entry. The supplied dictionary is not
    modified, and this function performs no model inference.
    """
    parameter_type = function_object.parameters[given_parameter_name].type

    if parameter_type == "string":
        system_instruction = (
            "Extract the value of the current parameter from the user's "
            "request. Use the function description and parameter name to "
            "determine which part of the request supplies that value. "
            "Distinguish the requested action from its argument. "
            "Do not return the command itself unless it is explicitly the "
            "requested argument value. "
            "Preserve the complete value exactly, including empty strings, "
            "spaces, punctuation, and Unicode characters. "
            "Quoted text may be an argument, but is not automatically the "
            "value of every parameter. "
            "If the request mentions both the target of an action and "
            "additional words describing how to perform that action, "
            "identify which one corresponds to the current parameter. "
            "For a parameter referring to the recipient or target, "
            "extract that recipient or target, not the words describing "
            "how the action should be performed. "
            "Return a JSON object. Do not execute the function."
        )
    else:
        system_instruction = (
            "Extract the input arguments for the selected function. "
            "Return a JSON object. Do not execute the function."
        )

    system_message = llm_helpers.format_request_message(
        "system", system_instruction
    )

    message = (
        f"Function: {function_object.name}\n"
        f"Description: {function_object.description}\n"
        f"User request: {user_request}\n"
    )

    message += f"Current parameter: {given_parameter_name}\n"
    message += "Required parameters:\n"
    message += f"Extracted parameters: {extracted_parameters}\n"

    for (
        parameter_name,
        parameter_value_type,
    ) in function_object.parameters.items():
        input_string = f"{parameter_name}: {parameter_value_type.type}"
        message += f"{input_string}\n"

    if function_object.parameters[given_parameter_name].type == "boolean":
        message += (
            "\nDetermine the desired final state represented by the "
            f"boolean parameter '{given_parameter_name}'. "
            "Pay careful attention to negation and which action it modifies. "
            "'Do not enable' and 'do not turn on' indicate an off state. "
            "'Do not disable' and 'do not turn off' indicate an on state. "
            "Map the desired state to true or false according to the "
            "meaning of the parameter name. "
            "Do not choose the value based on isolated action words."
        )

    if function_object.parameters[given_parameter_name].type == "number":
        message += (
            "\nIdentify the number corresponding to the current parameter "
            "using its name and the user's request. "
            "Preserve the sign of the number, including a leading minus sign. "
            "Previously extracted parameters are context, not values to copy. "
            "Different arguments may have the same numerical value.\n"
        )

    user_message = llm_helpers.format_request_message("user", message)

    json_text = json.dumps(
        extracted_parameters,
        ensure_ascii=False,
    )
    dictionary_prepare = json_text[:-1]
    json_key = json.dumps(given_parameter_name)
    if len(extracted_parameters) != 0:
        dictionary_prepare = dictionary_prepare + ","
    dictionary_prepare = dictionary_prepare + " " + f"{json_key}:"

    open_assistant_start = (
        "<|im_start|>assistant\n<think>\n\n</think>\n\n" + dictionary_prepare
    )
    return_message = system_message + user_message + open_assistant_start

    return return_message
