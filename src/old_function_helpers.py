from llm_sdk import Small_LLM_Model
from . import data_models, llm_helpers

import json


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
        selected_next_id = llm_helpers.select_next_token_id(
            token_logits, allowed_ids
        )
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
