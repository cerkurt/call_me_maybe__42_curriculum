"""Build selection prompts and generate text or constrained function names."""

from llm_sdk import Small_LLM_Model
from . import data_models


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
        list_element = f"{function.name}: {function.description}"
        available_functions.append(list_element)

    readable_list = "\n".join(available_functions)

    return (
        f"\nChoose the function that best matches the request.\n"
        f"Return only its exact name.\n\n"
        f"Available functions:: {readable_list}\n"
        f"\nRequest: {user_text}\n"
        f"Function: "
    )


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

    # Equal scores can belong to different IDs; enumerate tracks positions.
    for logit_index, logit in enumerate(logit_list):
        if logit_index not in allowed_token_ids:
            logit_list[logit_index] = neg_infinite

    next_allowed_logit = max(logit_list)
    allowed_token_index = logit_list.index(next_allowed_logit)

    return allowed_token_index


def generate_function_name(
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
