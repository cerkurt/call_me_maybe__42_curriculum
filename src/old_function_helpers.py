"""Retain earlier playground experiments outside the production pipeline.

These helpers explore unrestricted generation, JSON-string prefix
validation, vocabulary IDs, and legacy candidate maps. Some print debug
output. The active CLI uses the dedicated selector, extractor, regex,
shared LLM, and generation-flow modules instead.
"""

from llm_sdk import Small_LLM_Model
from . import data_models, llm_helpers

import json


def generate_text(
    model_object: Small_LLM_Model, start_text: str, max_token_count: int
) -> str:
    """Generate a greedy continuation for an early playground experiment.

    Args:
        model_object: Existing SDK model used for encoding and logits.
        start_text: Starting context to continue.
        max_token_count: Number of new token-selection steps, not words.

    Returns:
        The decoded continuation only, excluding the input context.

    Choose the largest logit at each step without candidate constraints or
    an early end-token check. Keep this experimental path separate from
    structured function-call generation.
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
    """Prepare a legacy plain-text prompt for extracting all arguments.

    Args:
        user_prompt: Natural-language request to include in the instruction.
        selected_function: Definition supplying its name, description, and
            required argument names/types.

    Returns:
        Instruction text ending with Arguments:, requesting a JSON object
        of argument values. It contains no chat-role wrappers, generated
        values, or token constraints by itself.
    """
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
    """Report the parsing state of an experimental JSON string prefix.

    Args:
        text_output: Decoded text to inspect from its beginning.

    Returns:
        One of not_started, in_string, backslash, unicode, completed, or
        invalid. Incomplete but syntactically plausible prefixes have their
        current state rather than completed. Invalid states remain invalid.

    Allow JSON whitespace before and after a string, track supported escapes
    and four hexadecimal digits after a Unicode escape, and reject raw
    control characters inside the string. This checks prefix syntax, not
    whether a value matches the request or a function's parameter schema.
    """
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
    """Check whether text can extend a syntactically plausible string prefix.

    Args:
        generated_text: Current decoded prefix, which must not be invalid.
        control_text: Nonempty decoded piece to test after that prefix.

    Returns:
        True when the combined text does not enter the invalid state; False
        for an invalid initial prefix, an empty piece, or an invalid result.
        A True result permits an incomplete prefix and does not require a
        completed JSON string.
    """
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
    """Filter token candidates by the decoded JSON-string prefix state.

    Args:
        model_object: Existing SDK model used to decode trial sequences.
        generated_token_ids: Flat list of IDs already generated for the value.
        candidate_token_ids: Token IDs to try as the next continuation.

    Returns:
        Candidate IDs whose complete trial decoding is nonempty and does
        not have the invalid string state. Preserve the candidates' order.

    Concatenate into a new trial list for each candidate; do not modify the
    generated IDs. This may decode many sequences per step and does not
    check the semantic meaning of the resulting text.
    """
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
    """Generate a complete JSON string in the legacy experiment.

    Args:
        model_object: Existing SDK model used for encoding, logits, and decode.
        order_text: Model prompt used as generation context.
        candidate_token_ids: Token IDs to filter by JSON-string prefix syntax.
        max_token_count: Maximum number of new tokens to select.

    Returns:
        The generated JSON string text, including its quotation marks and
        any accepted surrounding whitespace. The value is not json.loads()
        parsed here.

    Raises:
        ValueError: No next token is allowed or no complete string is reached
            within the token budget.

    Recompute valid next IDs, mask logits, and greedily append the chosen ID
    to both context and value histories. Print the decoded prefix and its
    state at each step. JSON-string syntax does not guarantee a correct
    argument or regex, and this path is not used by the production flow.
    """
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
    """Encode each available function name into a legacy candidate mapping.

    Args:
        model_object: Existing SDK model providing the tokenizer.
        validated_functions: FunctionDefinition instances supplying names.

    Returns:
        Function names mapped to flat token-ID lists. Select the first row
        of each encoded tensor and convert it with tolist(). Repeated names
        overwrite earlier entries in the resulting dictionary.
    """
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
    """Build the duplicate name-to-ID mapping retained from earlier tests.

    Args:
        model_object: Existing SDK model used to encode function names.
        validated_functions: Definitions supplying the candidate names.

    Returns:
        A dictionary mapping names to flat token-ID lists, with the same
        structure as function_name_encoder(). This helper remains for legacy
        playground calls rather than the active hybrid selection path.
    """
    name_id_match: dict = {}

    for function in validated_functions:
        encoded_name = model_object.encode(function.name)
        token_id_list: list = encoded_name[0].tolist()
        name_id_match.update({function.name: token_id_list})

    return name_id_match


def candidate_token_id_generator(model_object: Small_LLM_Model) -> list[int]:
    """Read token IDs from the vocabulary file exposed by the SDK.

    Args:
        model_object: Existing SDK model providing get_path_to_vocab_file().

    Returns:
        The vocabulary dictionary's values as a flat list in file order.
        These IDs are candidates for the legacy JSON-string experiments;
        the list need not cover every position in the model's logit vector.

    File and JSON decoding errors propagate to the caller. The context
    manager closes the vocabulary file after reading it.
    """
    candidate_token_ids: list = []
    file_path = model_object.get_path_to_vocab_file()
    with open(file_path, "r") as json_file:
        #  keys token texts, values are ids.
        file_content = json.load(json_file)
        for token_text, token_id in file_content.items():
            candidate_token_ids.append(token_id)

    return candidate_token_ids
