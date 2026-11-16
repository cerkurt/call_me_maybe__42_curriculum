from llm_sdk import Small_LLM_Model
from . import data_models, llm_helpers


def build_function_selector_prompt(
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
    system_message = llm_helpers.format_request_message(
        "system", message_to_model
    )
    user_message = llm_helpers.format_request_message("user", selection_prompt)
    open_assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    return system_message + user_message + open_assistant_start


def capability_score(
    model_object: Small_LLM_Model,
    user_request: str,
    function: data_models.FunctionDefinition,
) -> float:
    parameters = []

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

    return yes_score - no_score


def hybrid_function_selector(
    model_object: Small_LLM_Model,
    user_request: str,
    functions: list[data_models.FunctionDefinition],
) -> str:
    if not functions:
        raise ValueError("No available functions.")

    selection_prompt = build_function_selector_prompt(user_request, functions)

    best_function = None
    second_function = None

    best_score = float("-inf")
    second_score = float("-inf")

    for function in functions:
        score = llm_helpers.score_function_candidate(
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
