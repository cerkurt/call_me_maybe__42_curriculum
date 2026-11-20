"""Choose an available function using name scores and capability checks.

Build chat-formatted selection prompts, score complete candidate names,
and compare the two leading functions when their scores are close.
These helpers select a name without extracting or executing arguments.
"""

from llm_sdk import Small_LLM_Model

from . import data_models, llm_helpers


def build_function_selector_prompt(
    user_text: str,
    validated_functions: list[data_models.FunctionDefinition],
) -> str:
    """Format one request and the available function metadata for selection.

    Args:
        user_text: Original natural-language request to classify.
        validated_functions: Definitions supplying names, descriptions, and
            parameter names/types for the displayed function list.

    Returns:
        One string containing system and user messages followed by an open
        assistant response. The instruction requests only a function name
        and explains how to interpret requests containing negation.

    This function prepares text only; it does not call the model.
    """
    available_functions: list[str] = []

    for function in validated_functions:
        parameter_list: list[str] = []

        # Each definition is a model; its parameters map names to type models.
        for name, definition in function.parameters.items():
            parameter_list.append(f"{name}: {definition.type}")

        parameter = ", ".join(parameter_list)

        list_element = f"{function.name}({parameter}): {function.description}"

        available_functions.append(list_element)

    # Keep each function on its own line so its metadata stays together.
    readable_list_for_model = "\n".join(available_functions)

    function_selection_prompt = (
        f"\nChoose the function that best matches the request.\n"
        f"Return only its exact name.\n\n"
        f"Available functions: {readable_list_for_model}\n"
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
    system_message = llm_helpers.format_request_message(
        "system", message_to_model
    )
    user_message = llm_helpers.format_request_message(
        "user", function_selection_prompt
    )
    # Leave the assistant message open for the candidate name to follow.
    open_assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    return system_message + user_message + open_assistant_start


def should_reject_request(
    model_object: Small_LLM_Model,
    user_request: str,
    functions: list[data_models.FunctionDefinition],
) -> bool:
    """Identify requests that clearly lack a supported operation."""
    descriptions = "\n".join(
        f"{function.name}: {function.description}" for function in functions
    )

    instruction = (
        "Decide whether the user's message describes an identifiable task "
        "that at least one available function can perform. "
        "Answer No if the message is empty, meaningless, or does not identify "
        "a supported operation. Do not invent an operation for bare text "
        "or a standalone letter. A short request is valid when its intended "
        "operation is clear. Negative instructions can still identify a "
        "supported operation. Answer only Yes or No."
    )

    system_message = llm_helpers.format_request_message(
        "system",
        instruction + "\nAvailable functions:\n" + descriptions,
    )

    examples = (
        llm_helpers.format_request_message("user", "Request: 'zxq'")
        + llm_helpers.format_request_message("assistant", "No")
        + llm_helpers.format_request_message(
            "user",
            f"Request: {('Use ' + functions[0].name)!r}",
        )
        + llm_helpers.format_request_message("assistant", "Yes")
    )

    user_message = llm_helpers.format_request_message(
        "user", f"Request: {user_request!r}"
    )

    open_assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prompt = system_message + examples + user_message + open_assistant_start

    yes_score = llm_helpers.score_function_candidate(
        model_object, prompt, "Yes"
    )
    no_score = llm_helpers.score_function_candidate(model_object, prompt, "No")

    if no_score - yes_score >= 2.0:
        return True

    return False


def capability_score(
    model_object: Small_LLM_Model,
    user_request: str,
    function: data_models.FunctionDefinition,
) -> float:
    """Compare the model's Yes and No scores for one function's suitability.

    Args:
        model_object: Existing SDK model reused for both scoring calls.
        user_request: Request whose required capability is being assessed.
        function: Definition providing the name, description, and parameters.

    Returns:
        The complete-sequence log score of Yes minus that of No under the
        capability prompt. A larger value favors a Yes answer; the result
        is a score difference rather than a normalized confidence percentage.
    """
    parameters: list[str] = []

    for name, definition in function.parameters.items():
        parameters.append(f"{name}: {definition.type}")

    parameter_text = ", ".join(parameters)

    system_message = llm_helpers.format_request_message(
        "system",
        "Determine whether the given function can fulfill "
        "the user's request by choosing appropriate argument "
        "values. Consider the function's actual capability, "
        "not whether the user wants an action enabled or "
        "disabled. Answer only Yes or No.",
    )

    user_message = llm_helpers.format_request_message(
        "user",
        f"Request: {user_request}\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: {parameter_text}\n\n"
        "Can this function fulfill the request?",
    )

    assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    prompt = system_message + user_message + assistant_start

    yes_score = llm_helpers.score_function_candidate(
        model_object, prompt, "Yes"
    )

    no_score = llm_helpers.score_function_candidate(model_object, prompt, "No")

    # Compare Yes and No under the same prompt, not across different contexts.
    return yes_score - no_score


def hybrid_function_selector(
    model_object: Small_LLM_Model,
    user_request: str,
    functions: list[data_models.FunctionDefinition],
) -> str:
    """Select a function by name score, resolving close scores by capability.

    Args:
        model_object: Existing SDK model used for all scoring operations.
        user_request: Natural-language request to classify.
        functions: Available validated function definitions.

    Returns:
        An available function name. Return the leading candidate directly
        when it is the only scored candidate or leads by at least 2.0.
        Otherwise compare the two leaders' capability scores, favoring the
        original leader when those scores tie.

    Raises:
        ValueError: No functions are supplied or no best candidate is found.

    Rank names by the sum of their token log-probabilities. The 2.0 margin
    is a score threshold, not an accuracy percentage. Choosing an available
    name does not guarantee that its meaning matches every request.
    """
    if not functions:
        raise ValueError("No available functions to select.")

    function_selection_prompt = build_function_selector_prompt(
        user_request, functions
    )

    best_function = None
    second_function = None

    best_score = float("-inf")
    second_score = float("-inf")

    if should_reject_request(model_object, user_request, functions):
        raise ValueError(
            "The request does not clearly describe a supported operation."
        )

    for function in functions:
        # Score every token in the known name, not just its first token.
        score = llm_helpers.score_function_candidate(
            model_object,
            function_selection_prompt,
            function.name,
        )

        if score > best_score:
            # A new leader moves the previous leader into second place.
            second_function = best_function
            second_score = best_score

            best_function = function
            best_score = score

        elif score > second_score:
            # This candidate improves second place without replacing first.
            second_function = function
            second_score = score

    if best_function is None:
        raise ValueError("No function candidate could be scored.")

    if second_function is None:
        return best_function.name

    # A close name-score ranking triggers a separate capability comparison.
    margin = best_score - second_score

    if margin >= 2.0:
        return best_function.name

    first_capability = capability_score(
        model_object, user_request, best_function
    )

    second_capability = capability_score(
        model_object, user_request, second_function
    )

    # On a capability tie, retain the original name-score leader.
    if first_capability >= second_capability:
        return best_function.name

    return second_function.name
