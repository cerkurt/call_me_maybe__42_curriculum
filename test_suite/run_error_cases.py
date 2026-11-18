from pathlib import Path

import subprocess
import sys


class C:
    """ANSI color codes used to make terminal output easier to read."""

    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    PURPLE = "\033[35m"
    BG_RED = "\033[41m"


def run_error_cases() -> None:
    project_root = Path(__file__).resolve().parent.parent
    error_directory = project_root / "test_suite" / "01_errors"

    error_cases = [
        ("--functions_definition", "empty_functions.json"),
        ("--functions_definition", "missing_parameters.json"),
        ("--functions_definition", "unsupported_type.json"),
        ("--functions_definition", "duplicate_functions.json"),
        ("--functions_definition", "invalid_json.json"),
        ("--functions_definition", "wrong_root.json"),
        ("--functions_definition", "does_not_exist.json"),
        ("--input", "missing_prompt_field.json"),
        ("--input", "no_numbers_prompt.json"),
    ]

    for option, filename in error_cases:
        function_path = error_directory / "valid_functions.json"
        prompt_path = error_directory / "valid_prompts.json"

        if option == "--functions_definition":
            function_path = error_directory / filename
        else:
            prompt_path = error_directory / filename

        print(f"{C.BLUE}\n===== {filename} ====={C.RESET}", flush=True)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "src",
                "--functions_definition",
                str(function_path),
                "--input",
                str(prompt_path),
                "--output",
                str(project_root / "data/output/error_check.json"),
            ],
            cwd=project_root,
            check=False,
        )


if __name__ == "__main__":
    run_error_cases()
