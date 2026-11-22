"""Share chat formatting, token constraints, and candidate scoring helpers.

Use the SDK's public encoding and logit methods with a model supplied by
the caller. Keep token-ID positions intact when masking logits, and keep
generated candidate prefixes separate from their full model context.
"""

from llm_sdk import Small_LLM_Model

import math


def format_request_message(identity: str, content: str) -> str:
    """Wrap content in a completed chat message for the supplied role.

    Args:
        identity: Role text, such as system, user, or assistant.
        content: Message body to place between the chat markers.

    Returns:
        The role header, content, closing message marker, and final newline
        as one string. This helper does not validate the role or create an
        open assistant response.
    """
    return f"<|im_start|>{identity}\n" f"{content}<|im_end|>\n"


def find_allowed_next_token_ids(
    name_id_pair: dict[str, list[int]],
    candidate_id_list: list[int],
) -> list[int]:
    """Find unique token IDs that can extend the current candidate prefix.

    Args:
        name_id_pair: Candidate strings mapped to complete token-ID lists.
        candidate_id_list: IDs generated for the candidate so far, excluding
            the prompt IDs. An empty list represents the starting prefix.

    Returns:
        The next IDs from unfinished candidates whose complete initial slice
        matches the generated prefix. Return an empty list if none can
        continue. Deduplication uses a set, so output order is not guaranteed.

    Neither the candidate mapping nor the generated prefix is modified.
    """
    allowed_id_list: set[int] = set()
    for token_id in name_id_pair.values():
        # An empty generated prefix matches every candidate's empty slice.
        # Later, only candidates beginning with all generated IDs can remain.
        if (token_id[: len(candidate_id_list)] == candidate_id_list) and len(
            token_id
        ) != len(candidate_id_list):
            # After N generated IDs, the following candidate index is N.
            allowed_id_list.add(token_id[len(candidate_id_list)])

    # Shared prefixes can offer the same next ID; the set keeps it only once.
    return list(allowed_id_list)


def select_next_token_id(
    logit_list: list[float],
    allowed_token_ids: list[int],
) -> int:
    """Mask logits in place and return the index of the highest score.

    Args:
        logit_list: Next-token scores indexed by token ID; modified in place.
        allowed_token_ids: Permitted indexes into the score list.

    Returns:
        The index of the largest score after forbidden positions are replaced
        with negative infinity. Equal maxima favor the first index.

    Raises:
        ValueError: The allowed-ID list is empty.

    Callers must supply valid vocabulary indexes and usable allowed scores.
    Keeping all positions preserves the correspondence between score indexes
    and token IDs; this helper does not compact the score list.
    """
    neg_infinite = float("-inf")

    if len(allowed_token_ids) == 0:
        raise ValueError("There are no allowed token ids in the given list.")

    list_to_set = set(allowed_token_ids)

    # Equal scores can belong to different IDs; enumerate tracks positions.
    for logit_index, logit in enumerate(logit_list):
        if logit_index not in list_to_set:
            # Keep each ID's position; mask its score instead of removing it.
            logit_list[logit_index] = neg_infinite

    next_allowed_logit = max(logit_list)
    allowed_token_index = logit_list.index(next_allowed_logit)

    return allowed_token_index


def generate_name_from_candidates(
    model_object: Small_LLM_Model,
    selection_prompt: str,
    function_id_match: dict[str, list[int]],
    max_token_count: int,
) -> str:
    """Return an exact candidate using token-by-token logit constraints.

    Args:
        model_object: Existing SDK model reused throughout generation.
        selection_prompt: Model context for selecting the current value.
        function_id_match: Candidate strings mapped to their encoded ID lists.
            Despite the name, candidates may be argument values or option
            codes as well as function names.
        max_token_count: Maximum number of new candidate tokens to select.

    Returns:
        The original mapping key whose entire token sequence matches the
        generated IDs. A complete match is checked before each step and
        once more after the final permitted step.

    Raises:
        ValueError: No continuation is allowed or the token budget expires
            without a complete match.

    Keep prompt-plus-continuation IDs for model context and continuation-only
    IDs for prefix checks. At each step, mask forbidden scores, greedily
    select the largest allowed score, and append the selected ID to both
    histories. Candidate membership is enforced, not semantic correctness.
    """
    encode_result = model_object.encode(selection_prompt)
    # encode() returns a batch: take its first row to get a flat ID list.
    all_token_ids = encode_result[0].tolist()

    # Prefix checks use only the continuation, not the original prompt IDs.
    generated_token_ids: list[int] = []

    allowed_token_ids = find_allowed_next_token_ids(
        function_id_match, generated_token_ids
    )

    for _ in range(max_token_count):
        # Return the original candidate text once its whole ID list matches.
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


def score_function_candidate(
    model_object: Small_LLM_Model,
    selection_prompt: str,
    candidate_scoring_text: str,
) -> float:
    """Sum token log-probabilities for a complete candidate continuation.

    Args:
        model_object: Existing SDK model providing encoding and logits.
        selection_prompt: Context before the candidate continuation.
        candidate_scoring_text: Candidate text to score; may also be Yes or No.

    Returns:
        The sum of the candidate tokens' conditional log-probabilities.
        Longer candidates are not length-normalized. An empty encoded
        candidate has a total score of 0.0.

    Encode the prompt and candidate separately. For each candidate token,
    compute a stable log-sum-exp over the logits, add that token's log score,
    and append its known ID to the context for the next step. This scores a
    fixed continuation rather than choosing new tokens or masking logits.
    """
    context_ids = model_object.encode(selection_prompt)[0].tolist()
    candidate_ids = model_object.encode(candidate_scoring_text)[0].tolist()

    total_score = 0.0

    for token_id in candidate_ids:
        logits = model_object.get_logits_from_input_ids(context_ids)

        # Subtract the maximum before exp() to avoid very large exponentials.
        max_logit = max(logits)
        exp_sum = 0.0

        for logit in logits:
            exp_sum += math.exp(logit - max_logit)

        log_sum_exp = max_logit + math.log(exp_sum)

        # This is the current token's log-softmax score over the vocabulary.
        total_score += logits[token_id] - log_sum_exp
        # Append the known candidate token so its next token has full context.
        context_ids.append(token_id)

    return total_score
