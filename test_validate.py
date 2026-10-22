import json
from src import models, loader
from pydantic import ValidationError


class FunctionDefinitionTests:
    def print_valid_function(
        self, test_function: models.FunctionDefinition
    ) -> None:
        print(f"\nFunction name: {test_function.name}")
        print(f"Function description: {test_function.description}")

        parameters_dict = {}
        for key, value in test_function.parameters.items():
            parameters_dict.update({key: value.type})

        print(f"Function parameters: {parameters_dict}")
        print(f"Function returns: {test_function.returns.type}")


class PrompTests:
    def print_valid_prompt(
        self, number: int, test_prompt: models.PromptTest
    ) -> None:
        print(f"Prompt {number}: {test_prompt.prompt}")


def test() -> None:
    print("Testing the functios_definition.json load")
    try:
        test_validation = loader.model_validation(
            "data/input/functions_definition.json", models.FunctionDefinition
        )
        print(f"{len(test_validation)} items in loaded JSON list.")
        print(test_validation)
    except FileNotFoundError as e:
        print(
            f"[{type(e).__name__}] File not found in given path. "
            f"Check the file path and/or file name."
        )
    except json.JSONDecodeError as e:
        print(
            f"[{type(e).__name__}] Invalid JSON at line {e.lineno}, "
            f"column {e.colno}: {e.msg}"
        )
    except ValidationError as e:
        print(
            f"[{type(e).__name__}] Input validation failed. "
            f"Check the reported fields: {e}"
        )
    except ValueError as e:
        print(f"[{type(e).__name__}] Invalid input data {e}")
    except Exception as e:
        print(f"[{type(e).__name__}] {e}")

    print("\nTesting the function_calling_tests.json load")
    try:
        test_validation_1 = loader.model_validation(
            "data/input/function_calling_tests.json", models.PromptTest
        )
        print(f"{len(test_validation_1)} items in loaded JSON list.")
        print(test_validation_1)
    except FileNotFoundError as e:
        print(
            f"[{type(e).__name__}] File not found in given path. "
            f"Check the file path and/or file name."
        )
    except json.JSONDecodeError as e:
        print(
            f"[{type(e).__name__}] Invalid JSON at line {e.lineno}, "
            f"column {e.colno}: {e.msg}"
        )
    except ValidationError as e:
        print(
            f"[{type(e).__name__}] Input validation failed. "
            f"Check the reported fields: {e}"
        )
    except ValueError as e:
        print(f"[{type(e).__name__}] Invalid input data {e}")
    except Exception as e:
        print(f"[{type(e).__name__}] {e}")


if __name__ == "__main__":
    test()
