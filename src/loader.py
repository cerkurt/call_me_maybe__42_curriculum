import json
from pydantic import BaseModel


def load_json_file(path: str) -> list:
    with open(path, "r") as file_to_load:
        loaded_file = json.load(file_to_load)

    if not isinstance(loaded_file, list):
        raise ValueError("[load_json_file()] Loaded JSON data must be a list.")

    return loaded_file


def model_validation(path: str, model_class: type[BaseModel]) -> list:
    json_file = load_json_file(path)

    valid_list = []
    for list_element in json_file:
        if not isinstance(list_element, dict):
            raise ValueError(
                "[model_validation()] Elements of JSON list must be dict."
            )

        valid_data = model_class(**list_element)
        valid_list.append(valid_data)

    return valid_list
