import json
from src import models


def write_results(path: str, results: list[models.FunctionCallResult]) -> None:
    model_to_dict_list = []
    for model in results:
        result_dict = model.model_dump()
        model_to_dict_list.append(result_dict)

    with open(path, "w") as dummy_json:
        json.dump(model_to_dict_list, dummy_json, indent=4)


def test() -> None:
    path = "data/output/dummy.json"
    model_object = models.FunctionCallResult(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={"a": 2.0, "b": 3.0},
    )
    write_results(path, [model_object])


if __name__ == "__main__":
    test()
