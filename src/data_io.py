"""Load UTF-8 JSON arrays, validate input models, and serialize results.

Keep file handling separate from model inference. Input errors propagate
to callers; result serialization finishes before the output file opens.
"""

from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel

from . import data_models

import json

# Keep the returned list's model type tied to the class supplied by the caller.
TModel = TypeVar("TModel", bound=BaseModel)


class C:
    """ANSI color codes used to make terminal output easier to read."""

    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    PURPLE = "\033[35m"
    BG_RED = "\033[41m"


def load_json_file(path: str) -> list[object]:
    """Read a UTF-8 JSON file whose root must be an array.

    Args:
        path: Path to the input JSON file.

    Returns:
        The parsed array as a Python list. Elements are not validated here.

    Raises:
        ValueError: The parsed root is not a list.
        json.JSONDecodeError: The file does not contain valid JSON.
        OSError: The file cannot be opened or read.

    The context manager closes the file even when parsing fails.
    """
    with open(path, "r", encoding="utf-8") as file_to_load:
        loaded_file = json.load(file_to_load)

    # Valid JSON can still have the wrong root type for these input files.
    if not isinstance(loaded_file, list):
        raise ValueError("Loaded JSON data must be a list.")

    return loaded_file


def model_validation(path: str, model_class: type[TModel]) -> list[TModel]:
    """Load dictionary entries and construct the requested Pydantic models.

    Args:
        path: Path to a JSON array of objects.
        model_class: Pydantic model class used to validate each object.

    Returns:
        A list of model instances, preserving the input order.

    Raises:
        ValueError: An entry is not a dictionary. When model_class is exactly
            FunctionDefinition, also reject an empty list, blank function
            names, or duplicate names.
        pydantic.ValidationError: An entry violates the requested model.

    File and JSON parsing errors propagate from load_json_file(). Validation
    stops at the first error; no partial result list is returned.
    """
    json_file = load_json_file(path)

    valid_list: list[TModel] = []
    for list_element in json_file:

        # Check before ** unpacking, which requires a mapping of field names.
        if not isinstance(list_element, dict):
            raise ValueError("Elements of JSON list must be dict.")

        # ** passes dictionary keys and values as named model arguments.
        valid_data = model_class(**list_element)
        valid_list.append(valid_data)

    # These whole-list checks apply to definitions, not to every model type.
    if model_class is data_models.FunctionDefinition:
        if not valid_list:
            raise ValueError("Function list cannot be empty.")

        function_names = []

        for function in valid_list:
            if isinstance(function, data_models.FunctionDefinition):
                if not function.name.strip():
                    raise ValueError("Function name cannot be empty.")

                # Remember earlier names to reject ambiguous duplicate entries.
                if function.name in function_names:
                    raise ValueError(
                        f"Duplicate function name: "
                        f"{C.RED}{function.name}{C.RESET}"
                    )

                function_names.append(function.name)

    return valid_list


def write_results(
    path: str, results: list[data_models.FunctionCallResult]
) -> None:
    """Serialize result models as a UTF-8 JSON array and replace the file.

    Args:
        path: Destination file path. Missing parent directories are created.
        results: FunctionCallResult instances to convert with model_dump().

    Returns:
        None after writing the indented JSON array.

    Raises:
        ValueError: Serialization encounters a non-finite numeric value.
        TypeError: A dumped value cannot be serialized as JSON.
        OSError: A parent directory or the output file cannot be created or
            written.

    Prepare the complete text with json.dumps() before creating directories
    or opening the file. A serialization failure therefore leaves an existing
    file untouched. Successful writing replaces its previous contents and
    preserves literal Unicode characters with ensure_ascii=False.
    """
    model_to_dict_list: list[dict[str, object]] = []

    for model in results:
        # model_dump() also converts nested Pydantic objects into plain data.
        result_dict = model.model_dump()
        model_to_dict_list.append(result_dict)

    # Serialize first so invalid values cannot truncate an existing file.
    json_text = json.dumps(
        model_to_dict_list,
        ensure_ascii=False,
        allow_nan=False,
        indent=4,
    )

    # Output directories may be absent after a fresh clone or make clean.
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(json_text)
