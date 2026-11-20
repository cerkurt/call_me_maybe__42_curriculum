from pathlib import Path

import subprocess
import sys


from src import colors as C


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
