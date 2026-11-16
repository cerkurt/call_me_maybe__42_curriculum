#!/usr/bin/env python3

"""Pydantic schemas for function definitions, prompt inputs, and results."""

from pydantic import BaseModel
from typing import Literal

import math

# `functions_definition.json` → `list[FunctionDefinition]`
# `function_calling_tests.json` → `list[PromptTest]`
# Output JSON → `list[FunctionCallResult]`


class ParameterDefinition(BaseModel):
    """Declare the allowed JSON type for one function parameter."""

    type: Literal["number", "string", "boolean"]


class ReturnDefinition(BaseModel):
    """Declare the JSON type of a function's return value."""

    type: Literal["number", "string", "boolean"]


class FunctionDefinition(BaseModel):
    """Describe one available function and its parameter declarations.

    parameters is a dictionary: keys are parameter names and values are
    ParameterDefinition objects. Each value exposes its declared type
    through .type. The function's own name is stored in .name.
    """

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptTest(BaseModel):
    """Hold one user request from the input prompt array."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Hold a structured call with its original prompt and argument values.

    Parameter names and value types need separate checks against the
    selected FunctionDefinition; this model accepts arbitrary values.
    """

    prompt: str
    name: str
    parameters: dict[str, object]


def parameter_type_checker(
    function_object: FunctionDefinition, parameter_dict: dict
) -> None:
    parameters_from_function: list = []
    for parameter_name, parameter_type in function_object.parameters.items():
        parameters_from_function.append(parameter_name)

    parameters_from_dict: list = []
    for parameter_name, parameter_value in parameter_dict.items():
        parameters_from_dict.append(parameter_name)

    for parameter in parameters_from_function:
        if parameter not in parameters_from_dict:
            raise ValueError(
                f"Missing parameter ({parameter}) in generated parameter "
                f"dictionary."
            )

    for parameter in parameters_from_dict:
        if parameter not in parameters_from_function:
            raise ValueError(
                f"Unexpected parameter ({parameter}) found in in generated "
                f"parameter dictionary which is not part of function "
                f"parameters."
            )

    for parameter_name, parameter_type in function_object.parameters.items():
        parameter_value = parameter_dict[parameter_name]
        if (
            parameter_type.type == "number"
            and not isinstance(parameter_value, (int, float))
        ) or (
            parameter_type.type == "number"
            and isinstance(parameter_value, bool)
        ):
            raise ValueError(
                f"Unexpected parameter value type for parameter "
                f"'{parameter_name}' as '{parameter_value}'.\n"
                f"Current parameter value type is "
                f"{type(parameter_value).__name__} but must be part of 'int'"
                f" or 'float' class."
            )
        elif parameter_type.type == "number" and (
            isinstance(parameter_value, float)
            and not math.isfinite(parameter_value)
        ):
            raise ValueError(
                "Infinite number generated. Unsupported number to write in a "
                "standard JSON file."
            )
        elif parameter_type.type == "string" and not isinstance(
            parameter_value, str
        ):
            raise ValueError(
                f"Unexpected parameter value type for parameter "
                f"'{parameter_name}' as {parameter_value}'.\n"
                f"Current parameter value type is "
                f"{type(parameter_value).__name__} but must be part of 'str' "
                f"class."
            )

        elif parameter_type.type == "boolean" and not isinstance(
            parameter_value, bool
        ):
            raise ValueError(
                f"Unexpected parameter value type for parameter "
                f"'{parameter_name}' as '{parameter_value}'.\n"
                f"Current parameter value type is "
                f"{type(parameter_value).__name__} but must be part of 'bool'"
                f"class."
            )
