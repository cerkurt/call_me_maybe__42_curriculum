"""Define input and output schemas and validate extracted arguments.

Pydantic models describe functions, prompts, and result records. The
separate parameter checker compares actual argument names and values
with the selected function's declarations.
"""

from pydantic import BaseModel
from typing import Literal

import math


class C:
    """ANSI color codes used to make terminal output easier to read."""

    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    PURPLE = "\033[35m"
    BG_RED = "\033[41m"


class ParameterDefinition(BaseModel):
    """Declare the supported JSON type of one input argument.

    The type field accepts number, string, or boolean. This is metadata about
    an expected argument, not the value extracted from a user request.
    """

    type: Literal["number", "integer", "string", "boolean"]


class ReturnDefinition(BaseModel):
    """Declare the supported JSON type of a function's return value.

    The type field accepts number, string, or boolean. The application records
    function calls rather than executing them or producing this return value.
    """

    type: Literal["number", "integer", "string", "boolean"]


class FunctionDefinition(BaseModel):
    """Describe an available function and its declared input/output types.

    Store the function name and description alongside a parameter dictionary.
    Each parameter name maps to a ParameterDefinition instance whose type
    field declares the expected argument type. The returns field contains a
    ReturnDefinition instance.
    """

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptTest(BaseModel):
    """Hold the natural-language request from one input-array entry.

    The prompt field is a string. This model does not require a nonempty
    request or define an expected function-call result.
    """

    prompt: str


class FunctionCallResult(BaseModel):
    """Store an original request, selected function name, and argument values.

    The parameters dictionary holds extracted values, not ParameterDefinition
    objects. Its values are typed as object, so parameter_type_checker() must
    separately verify the selected function's required keys and value types.
    """

    prompt: str
    name: str
    parameters: dict[str, object]


def parameter_type_checker(
    function_object: FunctionDefinition, parameter_dict: dict[str, object]
) -> None:
    """Check argument keys and Python value types against a function schema.

    Args:
        function_object: Definition declaring the required argument names
            and their supported JSON types.
        parameter_dict: Extracted argument names mapped to actual values.

    Returns:
        None when every required key is present, no extra keys exist, and all
        values match their declarations. The dictionary is not modified.

    Raises:
        ValueError: A key is missing or unexpected, a value has the wrong
            type, or a number is a non-finite float.

    Number fields accept int or finite float values but reject bool, despite
    Python treating bool as an int subclass. String and boolean fields require
    str and bool respectively. This checks schema compliance, not whether a
    value correctly represents the user's intended argument.
    """
    parameters_from_function: list[str] = []
    for parameter_name, parameter_type in function_object.parameters.items():
        parameters_from_function.append(parameter_name)

    parameters_from_dict: list[str] = []
    for parameter_name, parameter_value in parameter_dict.items():
        parameters_from_dict.append(parameter_name)

    # Check both directions: required keys can be missing or extra keys added.
    for parameter in parameters_from_function:
        if parameter not in parameters_from_dict:
            raise ValueError(
                f"Missing parameter {C.RED}({parameter}){C.RESET} "
                f"in generated parameter dictionary."
            )

    for parameter in parameters_from_dict:
        if parameter not in parameters_from_function:
            raise ValueError(
                f"Unexpected parameter {C.RED}({parameter}){C.RESET} "
                f"found in generated parameter dictionary which is "
                f"not part of function parameters."
            )

    for parameter_name, parameter_type in function_object.parameters.items():
        # Compare declared type metadata with the extracted Python value.
        parameter_value = parameter_dict[parameter_name]
        # bool subclasses int, so number fields must reject it explicitly.
        if (
            parameter_type.type == "number"
            and not isinstance(parameter_value, (int, float))
        ) or (
            parameter_type.type == "number"
            and isinstance(parameter_value, bool)
        ):
            raise ValueError(
                f"Unexpected parameter value type for parameter "
                f"'{C.RED}{parameter_name}{C.RESET}' as "
                f"'{C.RED}{parameter_value}{C.RESET}'.\n"
                f"Current parameter value type is "
                f"{C.YELLOW}{type(parameter_value).__name__}{C.RESET} "
                f"but must be part of 'int' or 'float' class."
            )

        # NaN and infinities are floats, but not standard JSON numbers.
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
                f"'{C.RED}{parameter_name}{C.RESET}' as "
                f"'{C.RED}{parameter_value}{C.RESET}'.\n"
                f"Current parameter value type is "
                f"{C.YELLOW}{type(parameter_value).__name__}{C.RESET} "
                f"but must be part of 'str' class."
            )

        elif parameter_type.type == "boolean" and not isinstance(
            parameter_value, bool
        ):
            raise ValueError(
                f"Unexpected parameter value type for parameter "
                f"'{C.RED}{parameter_name}{C.RESET}' as "
                f"'{C.RED}{parameter_value}{C.RESET}'.\n"
                f"Current parameter value type is "
                f"{C.YELLOW}{type(parameter_value).__name__}{C.RESET} "
                f"but must be part of 'bool' class."
            )

        elif parameter_type.type == "integer" and (
            not isinstance(parameter_value, int)
            or isinstance(parameter_value, bool)
        ):
            raise ValueError(
                f"Parameter {parameter_name!r} must be an integer, "
                f"got {type(parameter_value).__name__}."
            )
