"""Run the command-line function-calling application.

Read paths from CLI options, validate both input files, create one SDK
model, generate structured calls, and write the resulting JSON array.
This module calls main() when loaded.
"""

from llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from pathlib import Path

from . import data_io, data_models, generation_flow
from . import colors as C

import argparse
import sys
import json
import traceback


def get_error_location(error: BaseException) -> str:
    """Return the source file, function, and line of an exception."""
    locations = traceback.extract_tb(error.__traceback__)

    if not locations:
        return "Unknown location"

    # The last traceback frame is where the exception was originally raised.
    last_location = locations[-1]
    file_name = Path(last_location.filename).name

    return (
        f"{C.YELLOW}File name:{C.RESET} {file_name}\n"
        f"{C.YELLOW}Function (location) name:{C.RESET} "
        f"{last_location.name}():\n"
    )


def print_status(message: str, success: bool = False) -> None:
    """Display an informational or success message immediately."""
    if success:
        label = f"{C.GREEN}[OK]{C.RESET}"
    else:
        label = f"{C.YELLOW}[INFO]{C.RESET}"

    # Print the status message to stderr, keeping it separate from standard
    # output. flush=True ensures the message is displayed immediately without
    # waiting for the output buffer to fill.
    print(
        f"{label} {message}",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    """Load the requested inputs, generate calls, and write the output file.

    Read --functions_definition, --input, and --output from the command line,
    using their configured defaults when omitted. Validate the input arrays
    before creating a single model shared by all requests. Pass the validated
    models to generation_flow and then serialize its results through data_io.

    Print contextual messages for handled loading, model initialization,
    generation, and writing failures, then exit with status 1. Successful
    execution returns None. Argument parsing handles help and invalid options.
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

    # Explicit CLI paths override defaults; omitted options keep defaults.
    args = parser.parse_args()

    # Update this before each stage so one handler can report what failed.
    flow_state = "loading function definitions"

    try:
        # Validate both files before spending time initializing the model.
        print_status("Loading and validating function definitions...")
        valid_function_list = data_io.model_validation(
            args.functions_definition, data_models.FunctionDefinition
        )

        flow_state = "loading user prompts"
        print_status("Loading and validating user prompts...")
        valid_prompt_list = data_io.model_validation(
            args.input, data_models.PromptTest
        )

        flow_state = "initializing the LLM"
        # This single object is reused for every prompt in the current run.
        print_status("Initializing language model...")
        model_object = Small_LLM_Model()

        flow_state = "generating function calls"
        print_status("Generating function calls...")
        model_generation_result = generation_flow.progress_builder(
            model_object,
            valid_function_list,
            valid_prompt_list,
        )

        flow_state = "writing JSON results"
        # Write only after generation returns the complete batch of results.
        print_status("Writing JSON results...")
        data_io.write_results(args.output, model_generation_result)

        print_status("Completed successfully!", success=True)

    except Exception as e:
        error_location = get_error_location(e)

        if isinstance(e, FileNotFoundError):
            error_message = f"File not found: {C.RED}{e.filename}{C.RESET}"

        # Check Decode/Pydantic errors before their ValueError parent class.
        elif isinstance(e, json.JSONDecodeError):
            error_message = (
                f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}"
            )

        elif isinstance(e, ValidationError):
            error_message = f"Input validation failed: {e}"

        elif isinstance(e, ValueError):
            error_message = str(e)

        elif isinstance(e, TypeError):
            error_message = f"Invalid data type: {e}"

        elif isinstance(e, OSError):
            error_message = f"File or operating system error: {e}"

        else:
            error_message = f"Unexpected error: {e}"
            traceback.print_exception(e)

        print(
            f"{C.BG_RED}[{type(e).__name__}]{C.RESET} "
            f"During {flow_state}, in:\n"
            f"{error_location}"
            f"{C.YELLOW}Error message:{C.RESET} {error_message}",
            file=sys.stderr,
        )
        # Tell the shell and Make that this run did not complete successfully.
        sys.exit(1)


if __name__ == "__main__":
    main()
