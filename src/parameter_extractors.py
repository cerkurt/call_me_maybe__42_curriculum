from . import data_models, llm_helpers

import re
import json
import math


def string_splitter_out_of_quotes(text: str) -> list[str]:
    completed_parts: list = []
    current_part: list = []
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
    completed_parts: list = []
    current_part: list = []
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
    inside_the_quotes_substrings = string_splitter_inside_the_quotes(
        user_request
    )
    out_of_the_quotes_substrings = string_splitter_out_of_quotes(user_request)

    word_filter_list: list = []
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

    # word_filter_list.append(user_request)
    for word in out_of_the_quotes_substrings:
        clean_word = word.rstrip(".!?")
        if clean_word != word and clean_word:
            word_filter_list.append(clean_word)

    word_filter_list.extend(inside_the_quotes_substrings)

    return list(dict.fromkeys(word_filter_list))


def extract_numbers_from_user_request(user_request: str) -> list[float]:
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
    extracted_parameters: dict,
) -> str:
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
