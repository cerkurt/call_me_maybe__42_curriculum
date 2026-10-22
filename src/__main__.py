import argparse
import sys
import json
from pydantic import ValidationError

from . import loader, models


def main() -> None:
    parser = argparse.ArgumentParser(description="Process some prompts")

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Input to be processed.",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Output result file path with the function calling results.",
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Definition of functions to be used during function calling.",
    )

    args = parser.parse_args()

    try:
        valid_function_list = loader.model_validation(
            args.functions_definition, models.FunctionDefinition
        )
    except FileNotFoundError as e:
        print(
            f"[{type(e).__name__}] File ({e.filename}) not "
            f"found in given path. "
            f"Check the file path and/or file name."
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(
            f"[{type(e).__name__}] Invalid JSON at line {e.lineno}, "
            f"column {e.colno}: {e.msg}"
        )
        sys.exit(1)
    except ValidationError as e:
        print(
            f"[{type(e).__name__}] Input validation failed. "
            f"Check the reported fields: {e}"
        )
        sys.exit(1)
    except ValueError as e:
        print(f"[{type(e).__name__}] Invalid input data at: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[{type(e).__name__}] {e}")
        sys.exit(1)

    try:
        valid_prompt_list = loader.model_validation(
            args.input, models.PromptTest
        )
    except FileNotFoundError as e:
        print(
            f"[{type(e).__name__}] File ({e.filename}) not "
            f"found in given path. "
            f"Check the file path and/or file name."
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(
            f"[{type(e).__name__}] Invalid JSON at line {e.lineno}, "
            f"column {e.colno}: {e.msg}"
        )
        sys.exit(1)
    except ValidationError as e:
        print(
            f"[{type(e).__name__}] Input validation failed. "
            f"Check the reported fields: {e}"
        )
        sys.exit(1)
    except ValueError as e:
        print(f"[{type(e).__name__}] Invalid input data at: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[{type(e).__name__}] {e}")
        sys.exit(1)

    print(len(valid_prompt_list))
    print(len(valid_function_list))


main()
