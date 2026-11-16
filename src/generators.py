"""Build selection prompts and generate text or constrained function names."""

from llm_sdk import Small_LLM_Model
from . import data_models

import re
import json
import math
import string


def generate_text(
    model_object: Small_LLM_Model, start_text: str, max_token_count: int
) -> str:
    """Continue start_text greedily using an already initialized model.

    max_token_count limits the number of new tokens, rather than words.
    Return the decoded continuation; the input prompt is excluded.
    """
    encode_result = model_object.encode(start_text)
    # encode() returns a batch; [0] selects its single token sequence.
    all_token_ids = encode_result[0].tolist()

    # Keep new IDs separately so the return value contains the continuation.
    generated_token_ids = []

    for _ in range(max_token_count):
        next_token_logits = model_object.get_logits_from_input_ids(
            all_token_ids
        )
        next_token_id = next_token_logits.index(max(next_token_logits))
        generated_token_ids.append(next_token_id)
        all_token_ids.append(next_token_id)

    decoded_token_ids: str = model_object.decode(generated_token_ids)

    return decoded_token_ids


def build_selection_prompt(
    user_text: str,
    validated_functions: list[data_models.FunctionDefinition],
) -> str:
    """Return instruction text containing the functions and one user request.

    Read names and descriptions from the supplied validated models.
    The returned string becomes the starting context for generation.
    """
    available_functions: list = []
    for function in validated_functions:
        parameters = ", ".join(
            f"{name}: {definition.type}"
            for name, definition in function.parameters.items()
        )

        list_element = f"{function.name}({parameters}): {function.description}"

        available_functions.append(list_element)

    readable_list = "\n".join(available_functions)

    selection_prompt = (
        f"\nChoose the function that best matches the request.\n"
        f"Return only its exact name.\n\n"
        f"Available functions:: {readable_list}\n"
        f"\nRequest: {user_text}\n"
        f"Function: "
    )
    message_to_model = (
        "Select the available function responsible for the operation "
        "or state described in the user's request. "
        "Match the subject and capability of the request to the "
        "function descriptions, not just individual action words. "
        "Negative instructions and prohibitions still refer to the "
        "operation or state they are about. "
        "Negation determines the intended argument values or final "
        "state; it should not by itself exclude the function that "
        "controls that state. "
        "Return only the exact function name. "
        "Do not produce arguments or execute the function."
    )
    system_message = format_request_message("system", message_to_model)
    user_message = format_request_message("user", selection_prompt)
    open_assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    return system_message + user_message + open_assistant_start


def find_allowed_next_token_ids(
    name_id_pair: dict, candidate_id_list: list
) -> list:
    """Return unique IDs that can continue the generated function-name prefix.

    name_id_pair maps each function name to its complete token-ID list.
    candidate_id_list holds only the name IDs produced so far; it starts
    empty and excludes the prompt IDs. Match this whole prefix against
    each candidate and collect the following ID from unfinished matches.

    Return an empty list when no candidate can continue. The set used for
    deduplication means the returned IDs have no guaranteed order.
    """
    allowed_id_list: set = set()
    for name, token_id in name_id_pair.items():
        if (token_id[: len(candidate_id_list)] == candidate_id_list) and len(
            token_id
        ) != len(candidate_id_list):
            # After N generated IDs, the following candidate index is N.
            allowed_id_list.add(token_id[len(candidate_id_list)])

    return list(allowed_id_list)


def select_next_token_id(logit_list: list, allowed_token_ids: list) -> int:
    """Mask scores in place and return the highest-scoring allowed token ID.

    A score's index is its token ID. allowed_token_ids must contain valid
    indexes into logit_list. Replace forbidden scores with negative
    infinity while keeping every score at its original position.

    Raise ValueError when the allowed-ID list is empty.
    """
    neg_infinite = float("-inf")

    if len(allowed_token_ids) == 0:
        raise ValueError("There are no allowed token ids in the given list")

    list_to_set = set(allowed_token_ids)
    # Equal scores can belong to different IDs; enumerate tracks positions.
    for logit_index, logit in enumerate(logit_list):
        if logit_index not in list_to_set:
            logit_list[logit_index] = neg_infinite

    next_allowed_logit = max(logit_list)
    allowed_token_index = logit_list.index(next_allowed_logit)

    return allowed_token_index


def generate_name_from_candidates(
    model_object: Small_LLM_Model,
    selection_prompt: str,
    function_id_match: dict,
    max_token_count: int,
) -> str:
    """Generate an exact candidate name using constraints at every token.

    Args:
        model_object: An already initialized SDK model to reuse.
        selection_prompt: Instruction text prepared for one user request.
        function_id_match: Names mapped to IDs from this model's tokenizer.
        max_token_count: Maximum number of newly generated name tokens.

    all_token_ids keeps the prompt and the growing name for model context.
    generated_token_ids keeps only the name prefix for candidate matching.
    Return the candidate's original name once its complete ID list matches.
    Raise ValueError if no continuation is allowed or the limit is reached
    without completing a candidate name.
    """
    encode_result = model_object.encode(selection_prompt)
    all_token_ids = encode_result[0].tolist()

    generated_token_ids: list = []

    allowed_token_ids = find_allowed_next_token_ids(
        function_id_match, generated_token_ids
    )

    for _ in range(max_token_count):
        for function_name, function_id in function_id_match.items():
            if function_id == generated_token_ids:
                return function_name

        # Recompute the choices using the prefix produced by earlier turns.
        allowed_token_ids = find_allowed_next_token_ids(
            function_id_match, generated_token_ids
        )
        logit_scores = model_object.get_logits_from_input_ids(all_token_ids)
        next_token_id = select_next_token_id(logit_scores, allowed_token_ids)
        # Both candidate matching and the model must see the selected ID.
        generated_token_ids.append(next_token_id)
        all_token_ids.append(next_token_id)

    # The final permitted turn may have just completed a valid name.
    for function_name, function_id in function_id_match.items():
        if function_id == generated_token_ids:
            return function_name
    # All candidates were checked; reaching here means none fully matched.
    raise ValueError(
        "Max token count overflow. Function name cannot be generated."
    )


def build_parameter_prompt(
    user_prompt: str, selected_function: data_models.FunctionDefinition
) -> str:
    prompt_name = user_prompt
    function_name = selected_function.name
    function_description = selected_function.description
    function_parameters = selected_function.parameters

    parameter_info_list: list = []
    for parameter_name, parameter_type in function_parameters.items():
        parameter_info_list.append(f"{parameter_name}: {parameter_type.type}")

    parameter_text = "\n".join(parameter_info_list)

    function_info_tag = (
        f"Extract the argument values from the user request for the "
        f"selected function. Listed types indicates the expections. "
        f"Extract the values from user request, do not use the type names as "
        f"values.\n"
        f"Get the parameter names from 'Required parameter(s).\n"
        f"Return a JSON object that contains only the parameters.\n"
        f"Do not add any description.\n\n"
        f"Prompt name: {prompt_name}\n"
        f"Function name: {function_name}\n"
        f"Function description: {function_description}\n"
        f"Required parameter(s):\n"
        f"{parameter_text}\n"
        f"Arguments: "
    )
    return function_info_tag


def get_json_string_state(text_output: str) -> str:
    current_state = "not_started"

    escape_chars = ['"', "\\", "/", "b", "f", "n", "r", "t"]

    hex_counter = 0
    low_case_hex = ["a", "b", "c", "d", "e", "f"]
    capital_case_hex = ["A", "B", "C", "D", "E", "F"]
    numbers_hex = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for char in text_output:
        if current_state == "not_started":
            if char == '"':
                current_state = "in_string"
            elif char in (" ", "\t", "\n", "\r"):
                current_state = "not_started"
            else:
                current_state = "invalid"

        elif current_state == "in_string":
            if char == '"':
                current_state = "completed"
            elif char == "\\":
                current_state = "backslash"
            elif (0 <= ord(char)) and (ord(char) <= 31):
                current_state = "invalid"

        elif current_state == "backslash":
            if char == '"':
                current_state = "in_string"
            elif char in escape_chars:
                current_state = "in_string"
            elif char == "u":
                current_state = "unicode"
                hex_counter = 4
            else:
                current_state = "invalid"

        elif current_state == "unicode":
            if (
                (char in numbers_hex)
                or (char in low_case_hex)
                or (char in capital_case_hex)
            ):
                hex_counter = hex_counter - 1
                if hex_counter == 0:
                    current_state = "in_string"
            else:
                current_state = "invalid"

        elif current_state == "completed":
            if char in (" ", "\t", "\n", "\r"):
                current_state = "completed"
            elif char:
                current_state = "invalid"

    return current_state


def is_this_allowed_string_piece(
    generated_text: str, control_text: str
) -> bool:
    if get_json_string_state(generated_text) == "invalid":
        return False
    elif control_text == "":
        return False

    new_string = generated_text + control_text
    result_string = get_json_string_state(new_string)

    if result_string != "invalid":
        return True
    else:
        return False


def find_allowed_string_token_ids(
    model_object: Small_LLM_Model,
    generated_token_ids: list,
    candidate_token_ids: list,
) -> list[int]:
    allowed_string_token_ids: list = []
    for token_id in candidate_token_ids:
        new_total_id_list = generated_token_ids + [token_id]
        decoded_ids_text = model_object.decode(new_total_id_list)
        if (
            get_json_string_state(decoded_ids_text) != "invalid"
            and decoded_ids_text != ""
        ):
            allowed_string_token_ids.append(token_id)

    return allowed_string_token_ids


def generate_json_string(
    model_object: Small_LLM_Model,
    order_text: str,
    candidate_token_ids: list,
    max_token_count: int,
) -> str:
    all_token_ids = model_object.encode(order_text)[0].tolist()
    generated_token_ids: list = []

    for _ in range(max_token_count):
        allowed_ids = find_allowed_string_token_ids(
            model_object, generated_token_ids, candidate_token_ids
        )
        token_logits = model_object.get_logits_from_input_ids(all_token_ids)
        selected_next_id = select_next_token_id(token_logits, allowed_ids)
        all_token_ids.append(selected_next_id)
        generated_token_ids.append(selected_next_id)
        generated_decoding = model_object.decode(generated_token_ids)
        print(
            repr(generated_decoding),
            get_json_string_state(generated_decoding),
        )
        if get_json_string_state(generated_decoding) == "completed":
            return generated_decoding
    raise ValueError(
        "Max token count overflow. JSON string cannot be generated."
    )


def format_request_message(identity: str, content: str) -> str:
    return f"<|im_start|>{identity}\n" f"{content}<|im_end|>\n"


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

    system_message = format_request_message("system", system_instruction)

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

    user_message = format_request_message("user", message)

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


def function_name_encoder(
    model_object: Small_LLM_Model,
    validated_functions: list[data_models.FunctionDefinition],
) -> dict:
    name_id_match: dict = {}
    for function in validated_functions:
        encoded_name = model_object.encode(function.name)
        function_name = function.name
        token_id_list: list = encoded_name[0].tolist()
        name_id_match.update({function_name: token_id_list})

    return name_id_match


def name_id_match_dict_generator(
    model_object: Small_LLM_Model,
    validated_functions: list[data_models.FunctionDefinition],
) -> dict:
    name_id_match: dict = {}

    for function in validated_functions:
        encoded_name = model_object.encode(function.name)
        token_id_list: list = encoded_name[0].tolist()
        name_id_match.update({function.name: token_id_list})

    return name_id_match


def candidate_token_id_generator(model_object: Small_LLM_Model) -> list[int]:
    candidate_token_ids: list = []
    file_path = model_object.get_path_to_vocab_file()
    with open(file_path, "r") as json_file:
        #  keys token texts, values are ids.
        file_content = json.load(json_file)
        for token_text, token_id in file_content.items():
            candidate_token_ids.append(token_id)

    return candidate_token_ids


def regex_candidate_generator(
    user_request: str, source_string: str
) -> dict[str, str]:
    literals = string_splitter_inside_the_quotes(user_request)

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
        format_request_message(
            "system",
            "Choose what the user wants to find and replace in the source "
            "text. Return only the letter of the best option. "
            "Do not execute the replacement.",
        )
        + format_request_message(
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

    selected_candidate = generate_name_from_candidates(
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

    substrings = string_splitter_out_of_quotes(user_request)

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


def score_function_candidate(
    model_object: Small_LLM_Model,
    selection_prompt: str,
    function_name: str,
) -> float:
    context_ids = model_object.encode(selection_prompt)[0].tolist()
    candidate_ids = model_object.encode(function_name)[0].tolist()

    total_score = 0.0

    for token_id in candidate_ids:
        logits = model_object.get_logits_from_input_ids(context_ids)

        max_logit = max(logits)
        log_sum_exp = max_logit + math.log(
            sum(math.exp(logit - max_logit) for logit in logits)
        )

        total_score += logits[token_id] - log_sum_exp
        context_ids.append(token_id)

    return total_score


def generate_name_by_full_scoring(
    model_object,
    selection_prompt: str,
    function_id_match: dict,
) -> str:

    scores: dict[str, float] = {
        function_name: score_function_candidate(
            model_object,
            selection_prompt,
            function_name,
        )
        for function_name in function_id_match
    }

    return max(scores, key=lambda name: scores[name])


def capability_score(
    model_object: Small_LLM_Model,
    user_request: str,
    function: data_models.FunctionDefinition,
) -> float:
    parameters = []

    for name, definition in function.parameters.items():
        parameters.append(f"{name}: {definition.type}")

    parameter_text = ", ".join(parameters)

    system_message = format_request_message(
        "system",
        "Determine whether the given function can fulfill "
        "the user's request by choosing appropriate argument "
        "values. Consider the function's actual capability, "
        "not whether the user wants an action enabled or "
        "disabled. Answer only Yes or No.",
    )

    user_message = format_request_message(
        "user",
        f"Request: {user_request}\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: {parameter_text}\n\n"
        "Can this function fulfill the request?",
    )

    assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    prompt = system_message + user_message + assistant_start

    yes_score = score_function_candidate(model_object, prompt, "Yes")

    no_score = score_function_candidate(model_object, prompt, "No")

    return yes_score - no_score


def hybrid_function_selection(
    model_object: Small_LLM_Model,
    user_request: str,
    functions: list[data_models.FunctionDefinition],
) -> str:
    if not functions:
        raise ValueError("No available functions.")

    selection_prompt = build_selection_prompt(user_request, functions)

    best_function = None
    second_function = None

    best_score = float("-inf")
    second_score = float("-inf")

    for function in functions:
        score = score_function_candidate(
            model_object,
            selection_prompt,
            function.name,
        )

        if score > best_score:
            second_function = best_function
            second_score = best_score

            best_function = function
            best_score = score

        elif score > second_score:
            second_function = function
            second_score = score

    if best_function is None:
        raise ValueError("No function candidate could be scored.")

    if second_function is None:
        return best_function.name

    margin = best_score - second_score

    if margin >= 2.0:
        return best_function.name

    first_capability = capability_score(
        model_object, user_request, best_function
    )

    second_capability = capability_score(
        model_object, user_request, second_function
    )

    if first_capability >= second_capability:
        return best_function.name

    return second_function.name


def progress_builder(
    model_object: Small_LLM_Model,
    validated_function_list: list[data_models.FunctionDefinition],
    validated_prompt_list: list[data_models.PromptTest],
) -> list[data_models.FunctionCallResult]:
    list_to_write: list = []

    for prompt in validated_prompt_list:
        user_request = prompt.prompt

        result_function_name = hybrid_function_selection(
            model_object,
            user_request,
            validated_function_list,
        )
        for function in validated_function_list:
            if function.name == result_function_name:
                regex_fields = {"source_string", "regex", "replacement"}

                is_regex_function = regex_fields.issubset(
                    function.parameters
                ) and all(
                    function.parameters[field].type == "string"
                    for field in regex_fields
                )

                parameter_dict: dict = {}
                for (
                    parameter_name,
                    parameter_type,
                ) in function.parameters.items():
                    candidates: list[float] | list[str] | list[bool]

                    if is_regex_function and parameter_name == "regex":
                        continue

                    if parameter_type.type == "number":
                        candidates = extract_numbers_from_user_request(
                            user_request
                        )
                        if not candidates:
                            raise ValueError(
                                f"No candidates found for parameter "
                                f"'{parameter_name}' in user request as "
                                f"{user_request}"
                            )

                    elif parameter_type.type == "string":
                        candidates = string_slicer(user_request)

                    elif parameter_type.type == "boolean":
                        candidates = [True, False]

                    else:
                        raise ValueError(
                            f"Unsupported parameter type for parameter "
                            f"'{parameter_name}' as '{parameter_type.type}'"
                        )

                    model_input_text = message_for_parameter_extraction(
                        user_request,
                        function,
                        parameter_name,
                        parameter_dict,
                    )

                    json_and_id_match_dict: dict = {}
                    for candidate in candidates:
                        json_text = " " + json.dumps(
                            candidate, ensure_ascii=False
                        )
                        id_list = model_object.encode(json_text)[0].tolist()
                        json_and_id_match_dict.update({json_text: id_list})

                    result = generate_name_from_candidates(
                        model_object,
                        model_input_text,
                        json_and_id_match_dict,
                        50,
                    )

                    parameter_value = json.loads(result)
                    parameter_dict[parameter_name] = parameter_value

                if is_regex_function:
                    source_string_value = parameter_dict["source_string"]
                    replacement_value = parameter_dict["replacement"]

                    if not isinstance(
                        source_string_value, str
                    ) or not isinstance(replacement_value, str):
                        raise ValueError(
                            "Regex source_string and replacement "
                            "must both be strings."
                        )

                    replacement = convert_regex_replacement_value(
                        user_request, replacement_value
                    )
                    parameter_dict["replacement"] = replacement

                    new_regex = generate_regex_pattern(
                        model_object,
                        user_request,
                        source_string_value,
                        replacement_value,
                    )
                    parameter_dict["regex"] = new_regex

                data_models.parameter_type_checker(function, parameter_dict)

                parameter_dict = {
                    parameter_name: parameter_dict[parameter_name]
                    for parameter_name in function.parameters
                }
                new_function_object = data_models.FunctionCallResult(
                    prompt=user_request,
                    name=function.name,
                    parameters=parameter_dict,
                )
                list_to_write.append(new_function_object)
                break

    return list_to_write
