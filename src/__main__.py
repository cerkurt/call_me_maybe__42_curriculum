"""Command-line entry point for loading and validating project inputs."""

import argparse
import sys
import json
from pydantic import ValidationError

from . import data_io, data_models, generators
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Parse paths, validate both input arrays, and print their item counts.

    Display loader errors at this command-line boundary and exit with
    status 1 on failure. The output path is parsed for later integration
    with generation and result writing.
    """
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
        valid_function_list = data_io.model_validation(
            args.functions_definition, data_models.FunctionDefinition
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
        valid_prompt_list = data_io.model_validation(
            args.input, data_models.PromptTest
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
        model_object = Small_LLM_Model()
    except ValueError as e:
        print(
            f"[{type(e).__name__}] Error generating model object in main. {e}"
        )
        sys.exit(1)
    except OSError as e:
        print(
            f"[{(type(e).__name__)}] Error with operating system during "
            f"generating model object in main. {e}"
        )
        sys.exit(1)

    try:
        model_generation_result = generators.progress_builder(
            model_object, valid_function_list, valid_prompt_list
        )
    except ValueError as e:
        print(f"[{type(e).__name__}] Error during model progress flow. {e}")
        sys.exit(1)

    try:
        data_io.write_results(args.output, model_generation_result)
    except OSError as e:
        print(
            f"[{type(e).__name__}] Error during output writing at "
            f"{args.output}. {e}"
        )
        sys.exit(1)


main()
