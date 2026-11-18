# CLI Test Suite

This suite runs JSON fixtures through the project's own command-line
application. A small comparison script checks saved results against
expected calls. No pytest setup is required for this workflow.

Run all commands from the project root, next to `pyproject.toml` and the
`Makefile`.

## Test groups

| Directory | Contents | Purpose |
|---|---|---|
| `01_errors` | Invalid inputs and two valid supporting files | Inspect error handling |
| `02_prompts_21` | Function definitions, 21 prompts, and expected results | Check the main generation flow |
| `03_prompts_42` | Function definitions, 42 prompts, and expected results | Exercise more formats and edge cases |

The two generation groups share the same eight function definitions.
The 42-prompt group includes the first 21 prompts and adds 21 more.

- `functions.json` describes the available functions and argument types.
- `prompts.json` contains the requests to process.
- `expected.json` contains reference calls for comparison. It is not an
  input file for the model.

## 1. Run error cases

```bash
make test-errors
```

This runs `test_suite/run_error_cases.py`. For each case, the script starts
the production CLI with a function file and a prompt file. It prints the
case filename followed by the application's messages. Each CLI invocation
is a separate process, so one failed invocation does not stop the others.

| File | Input option | Expected error |
|---|---|---|
| `empty_functions.json` | `--functions_definition` | `ValueError`: empty function list |
| `missing_parameters.json` | `--functions_definition` | `ValidationError`: missing `parameters` field |
| `unsupported_type.json` | `--functions_definition` | `ValidationError`: unsupported `banana` type |
| `duplicate_functions.json` | `--functions_definition` | `ValueError`: duplicate function name |
| `invalid_json.json` | `--functions_definition` | `JSONDecodeError`: trailing comma |
| `wrong_root.json` | `--functions_definition` | `ValueError`: object instead of an array |
| `does_not_exist.json` | `--functions_definition` | `FileNotFoundError` |
| `missing_prompt_field.json` | `--input` | `ValidationError`: missing `prompt` field |
| `no_numbers_prompt.json` | `--input` | Generation-stage `ValueError`: no number candidates |

`does_not_exist.json` is intentionally absent. The valid supporting files,
`valid_functions.json` and `valid_prompts.json`, let each case focus on one
problem.

The no-number case loads the LLM. The other cases are rejected before model
initialization. Inspect the messages: the script runs the cases but does
not automatically verify their error types or mark them as passing.
Completing the script does not by itself prove correct error handling.

The application still stops at the first error in each invocation. Error
fixtures are separate files so every problem can be inspected.

## 2. Run the 21-prompt group

```bash
make test-21
make compare-21
```

The first command runs generation and writes `data/output/results_21.json`.
The second compares that saved output with
`test_suite/02_prompts_21/expected.json` without loading the model again.

The prompts cover function selection, number/string/boolean arguments,
negative and decimal numbers, multiword values, Unicode, and regex
replacement requests.

## 3. Run the 42-prompt group

```bash
make test-42
make compare-42
```

The generated calls are written to `data/output/results_42.json` and
compared with `test_suite/03_prompts_42/expected.json`.

Additional cases include scientific notation, thousands separators,
decimals such as `.5`, multiword names, empty strings, embedded quotes and
apostrophes, Japanese text, boolean negation, and literal regex
metacharacters.

To compare only after generation succeeds, use:

```bash
make test-21 && make compare-21
make test-42 && make compare-42
```

Each generation command initializes one model and reuses it for that
group's prompts. The two groups are separate runs.

## Reading the comparison

`compare_results.py` reads both JSON arrays and compares calls in order.
For each expected call it prints `PASS`, `FAIL`, or `MISSING`. A failed
comparison shows the expected and actual calls. Extra actual records are
also displayed. The script continues after mismatches and ends with
correct/expected counts, missing/extra counts, and accuracy.

This uses Python equality and `assert`, with each `AssertionError` caught
inside the loop. Differences do not produce a nonzero exit status. Missing
files or malformed JSON still stop the script with an error. Read the
printed summary rather than relying on Make's exit status for accuracy.

The comparison has these limits:

- JSON indentation and object key order do not affect equality.
- Prompt order, original prompt text, function names, and argument values
  must match. Missing or extra fields cause a mismatch.
- Python considers `2` and `2.0` equal, but also considers `True` and `1`
  equal. This script does not replace parameter type validation in the
  application.
- Regex patterns are compared as strings. Different patterns with
  equivalent behavior need manual review.
- Accuracy is correct expected calls divided by expected calls. Extra
  records are reported separately and must also be checked.

The application extracts arguments; it does not execute the selected
function. For `Reverse 'hello'`, the argument is `hello`, not `olleh`.
An addition request should return its operands, not their sum.

## Output files and paths

Generation must finish successfully before its output is treated as a new
result. A failed run can leave an older output file in place. The comparison
script reads existing files and cannot tell when they were generated.
`make clean` removes `data/output/`, including saved test results.

These fixtures do not guarantee that every model answer will be correct.
A generation error may stop a group before an output file is written.
Such a run is not a completed evaluation.

The Makefile passes paths to the existing CLI. For example, the 21-prompt
group can also be run directly:

```bash
uv run python -m src \
  --functions_definition test_suite/02_prompts_21/functions.json \
  --input test_suite/02_prompts_21/prompts.json \
  --output data/output/results_21.json
```

Relative paths are resolved from the current working directory: `./` means
the current directory and `../` means its parent. If the fixtures are moved,
update the paths in the Makefile and the error runner as appropriate.
