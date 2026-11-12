"""Read JSON inputs, validate their objects, and write model results."""

import json
from pydantic import BaseModel
from . import data_models


def load_json_file(path: str) -> list:
    """Read a JSON array from path and return its contents as a Python list.

    The with block closes the file even when parsing fails. File errors
    and JSONDecodeError propagate to the caller. Raise ValueError if the
    parsed root value is not a list.
    """
    with open(path, "r") as file_to_load:
        loaded_file = json.load(file_to_load)

    if not isinstance(loaded_file, list):
        raise ValueError("[load_json_file()] Loaded JSON data must be a list.")

    return loaded_file


def model_validation(path: str, model_class: type[BaseModel]) -> list:
    """Load JSON dictionaries and return instances of the supplied model.

    model_class is a Pydantic model class, such as FunctionDefinition or
    PromptTest. Construct one instance for each dictionary in the array.
    Raise ValueError for non-dictionary elements and propagate model
    validation errors. The first error stops loading; no partial list
    is returned.
    """
    json_file = load_json_file(path)

    valid_list = []
    for list_element in json_file:
        if not isinstance(list_element, dict):
            raise ValueError(
                "[model_validation()] Elements of JSON list must be dict."
            )

        # ** passes dictionary keys and values as named model arguments.
        valid_data = model_class(**list_element)
        valid_list.append(valid_data)

    return valid_list


def write_results(
    path: str, results: list[data_models.FunctionCallResult]
) -> None:
    """Write result models as one indented JSON array, replacing the file.

    model_dump() converts a Pydantic instance to a Python dictionary.
    json.dump() writes the collected dictionaries to the opened file.
    The parent directory must already exist; file errors reach the caller.
    """
    model_to_dict_list = []
    for model in results:
        result_dict = model.model_dump()
        model_to_dict_list.append(result_dict)

    json_text = json.dumps(model_to_dict_list, indent=4, allow_nan=False)
    with open(path, "w") as dummy_json:
        dummy_json.write(json_text)
