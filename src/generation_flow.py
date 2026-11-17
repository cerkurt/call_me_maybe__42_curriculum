"""Coordinate function selection and typed argument extraction per request.

Reuse a caller-provided SDK model and validated inputs to build result
models. File loading, model construction, and output writing remain the
responsibility of the caller.
"""

from llm_sdk import Small_LLM_Model
from . import (
    data_models,
    llm_helpers,
    parameter_extractors as param_ext,
    function_selectors,
    regex_helpers,
)

import json


def progress_builder(
    model_object: Small_LLM_Model,
    validated_function_list: list[data_models.FunctionDefinition],
    validated_prompt_list: list[data_models.PromptTest],
) -> list[data_models.FunctionCallResult]:
    """Build structured function-call results for the supplied prompt models.

    Args:
        model_object: Existing SDK model shared by selection and extraction.
        validated_function_list: Available function definitions and schemas.
        validated_prompt_list: PromptTest instances to process in order.

    Returns:
        Collected FunctionCallResult instances in request order. Each result
        retains its original prompt and a parameter dictionary ordered like
        the selected definition.

    For each request, select a function with the hybrid selector and extract
    its arguments from typed candidate pools. Numbers come from the request,
    strings from the string slicer, and booleans from True/False candidates.
    Serialize candidates with a leading space and use token constraints to
    select each value, passing earlier arguments as extraction context.

    Definitions with source_string, regex, and replacement all declared as
    string use the regex helpers after source/replacement extraction. Skip
    ordinary extraction of regex, normalize supported replacement names, and
    then select a pattern. Check all keys and types before collecting a result.

    Raises:
        ValueError: Selection or constrained generation fails, no numeric
            candidate exists, a parameter type is unsupported, or final
            argument validation fails. Helper errors propagate to the caller.

    An empty prompt list returns an empty list. The function performs no file
    I/O and does not execute the selected application functions.
    """
    list_to_write: list = []

    for prompt in validated_prompt_list:
        user_request = prompt.prompt

        result_function_name = function_selectors.hybrid_function_selector(
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
                        candidates = (
                            param_ext.extract_numbers_from_user_request(
                                user_request
                            )
                        )
                        if not candidates:
                            raise ValueError(
                                f"No candidates found for parameter "
                                f"'{parameter_name}' in user request as "
                                f"{user_request}"
                            )

                    elif parameter_type.type == "string":
                        candidates = param_ext.string_slicer(user_request)

                    elif parameter_type.type == "boolean":
                        candidates = [True, False]

                    else:
                        raise ValueError(
                            f"Unsupported parameter type for parameter "
                            f"'{parameter_name}' as '{parameter_type.type}'"
                        )

                    model_input_text = (
                        param_ext.message_for_parameter_extraction(
                            user_request,
                            function,
                            parameter_name,
                            parameter_dict,
                        )
                    )

                    json_and_id_match_dict: dict = {}
                    for candidate in candidates:
                        json_text = " " + json.dumps(
                            candidate, ensure_ascii=False
                        )
                        id_list = model_object.encode(json_text)[0].tolist()
                        json_and_id_match_dict.update({json_text: id_list})

                    result = llm_helpers.generate_name_from_candidates(
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

                    replacement = (
                        regex_helpers.convert_regex_replacement_value(
                            user_request, replacement_value
                        )
                    )
                    parameter_dict["replacement"] = replacement

                    new_regex = regex_helpers.generate_regex_pattern(
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
