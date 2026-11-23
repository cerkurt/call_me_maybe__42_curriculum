"""Share chat formatting, cached logits, token constraints, and scoring.

Use the SDK's public encoding and logit methods with a model supplied by
the caller. Keep token-ID positions intact when masking logits, and keep
generated candidate prefixes separate from their full model context.

Reuse raw logits when the same unchanged model receives identical input
IDs. During constrained generation, skip inference for a forced next token.
Full candidate scoring still sums conditional log-probabilities over the
complete vocabulary; it does not mask logits or execute functions.
"""

from llm_sdk import Small_LLM_Model

from functools import lru_cache

import math


# Keep at most eight model/input combinations. When full, evict the entry
# that was least recently used. This cache exists only in runtime memory.
@lru_cache(maxsize=8)
def get_cached_logits(
    model_object: Small_LLM_Model,
    input_ids: tuple[int, ...],
) -> tuple[float, ...]:
    """Return next-token logits, reusing results for identical inputs.

    Args:
        model_object: Existing SDK model whose weights and settings remain
            unchanged while its cached results are used.
        input_ids: Complete context as a tuple of token IDs. Tuples can be
            cache keys; mutable lists cannot.

    Returns:
        An immutable tuple of raw logits, indexed by vocabulary token ID.
        These are not probabilities or already masked scores.

    The cache key includes both the model object and the complete input-ID
    tuple. The decorator reuses a stored result without entering this body
    when both arguments match. A changed context requires a new inference
    unless that exact model/context combination is still cached.

    Callers that mask logits must first copy the tuple into a list. If the
    model's weights or inference settings change, clear the cache before
    reusing this helper. The cache does not persist between program runs.
    """
    # The SDK expects a list. Freeze its output so masking cannot modify the
    # shared cached result through one of the callers.
    return tuple(model_object.get_logits_from_input_ids(list(input_ids)))


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

    This helper only checks prefix compatibility. It does not call the
    model, compare scores, choose a winner, or delete candidates. Neither
    the candidate mapping nor the generated prefix is modified. Completed
    candidates have no next ID; the caller checks complete matches.
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
            # Permission determines masking, even for a high-scoring token.
            logit_list[logit_index] = neg_infinite

    # Multiple IDs can remain allowed; choose just one for this step.
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
        max_token_count: Maximum number of new candidate tokens to select,
            including forced tokens. This is not a model-call limit.

    Returns:
        The original mapping key whose entire token sequence matches the
        generated IDs. A complete match is checked before each step and
        once more after the final permitted step.

    Raises:
        ValueError: No continuation is allowed or the token budget expires
            without a complete match.

    Keep prompt-plus-continuation IDs for model context and continuation-only
    IDs for prefix checks. If exactly one next ID is allowed, take it without
    model inference. Otherwise copy cached raw logits, mask forbidden IDs,
    and greedily select the largest allowed score. Append either kind of
    selected ID to both histories.

    Stop on a complete token-sequence match, not merely when one candidate
    remains. Return its original mapping key, so SDK decoding is unnecessary.
    Candidate membership is enforced, not semantic correctness.
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
        # A single remaining candidate may still need several more tokens.
        for function_name, function_id in function_id_match.items():
            if function_id == generated_token_ids:
                return function_name

        # Recompute the choices using the prefix produced by earlier turns.
        allowed_token_ids = find_allowed_next_token_ids(
            function_id_match, generated_token_ids
        )

        if len(allowed_token_ids) == 1:
            # Several candidates can share this one forced next ID. There
            # is no score comparison to make, so skip the model call.
            next_token_id = allowed_token_ids[0]

        else:
            # Masking changes a list in place. Copy the immutable cached
            # tuple so future callers can reuse the original raw logits.
            logit_scores = list(
                get_cached_logits(model_object, tuple(all_token_ids))
            )
            next_token_id = select_next_token_id(
                logit_scores, allowed_token_ids
            )

        # Record forced tokens too: later model calls receive the complete
        # context, while prefix checks see only the generated continuation.
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
    get cached or freshly computed raw logits, normalize over the entire
    vocabulary using log-sum-exp, and add that token's log-probability.
    Append the known candidate ID so the next score includes earlier tokens.

    Each call starts with the supplied prompt context. Identical contexts
    can reuse logits across candidates, including their shared prefixes or
    the Yes/No answers under the same question. The formula is unchanged.

    Only the supplied candidate tokens are scored. Their summed log-score
    corresponds to multiplying their conditional probabilities; it is not
    a correctness percentage. This helper scores a fixed continuation and
    does not choose tokens, mask logits, or execute the described function.
    """
    context_ids = model_object.encode(selection_prompt)[0].tolist()
    candidate_ids = model_object.encode(candidate_scoring_text)[0].tolist()

    total_score = 0.0

    for token_id in candidate_ids:
        # Matching requires the full context, not just the candidate name.
        # The cached tuple is safe here because scoring only reads logits.
        logits = get_cached_logits(model_object, tuple(context_ids))

        # Subtract the maximum before exp() to avoid very large exponentials.
        max_logit = max(logits)
        exp_sum = 0.0

        # Include every vocabulary logit, not only the current candidates.
        for logit in logits:
            exp_sum += math.exp(logit - max_logit)

        log_sum_exp = max_logit + math.log(exp_sum)

        # This is the current token's log-softmax score over the vocabulary.
        total_score += logits[token_id] - log_sum_exp
        # Append the known candidate token so its next token has full context.
        context_ids.append(token_id)

    return total_score
