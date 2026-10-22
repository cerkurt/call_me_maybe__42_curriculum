# call me maybe

## description
What is Function Calling?
A communication bridge between human language and LLM understanding by
translating human lanugage into structured call with typed arguments.

The Challange?
Using the constrained decoding method, guide the output of the small language
model token-by-token and ensure structured and valid output.

Mandatory part summary?
The task is to prepare a tool that helps translating natural language into
stuctured function calls.for example when asked the sum of 40 and 2, instead of
returning the result as output, it should produce a json file that contains
related function name and parameters that are included. must be implemented by
using constrained decoding.

```bash
function name: fn_add_numbers
parameters: {"a": 40, "b": 2}
```


## how to run
Default run:

```bash
uv run python -m src
```

## input files
The program reads two input files. (must be located at data/input/ directry)
    1. function_calling_test.json: this file includes natural language user
    prompts to be distiled as function calling results.
    2. function_definition.json: this file contains the aviable functions that
    the system can call and use. each function has a name, description,
    parameters and return type.


## output file format
This program writes one JSON array.
Each array must contain:
 - prompt: original human-language prompt
 - name: name of the selected function
 - parameters: all the required arguments for that call

No extra elements are wanted.
Example:

```
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "mame": "fn_add_numbers,
        "parameters": {
            "a": 2,
            "b": 3
        }
    }
]
```

## rules and constraints
- Python 3.10+
- flake8
- mypy .
- type hints
- docstrings (comment strings are optional but nice to have)
- Pydantic validation for all classes
- numpy and json are allowed
- dspy, pytorch, huggingface, transformers, outlines are forbidden
- Qwen/Qwen3-0.6B is the default model
- Function selection must be done using the LLM
- No heuristic function selection
- No privarte llm_sdk methods
- Graceful arror handling
- No hardcoding
- DO NOT HARDCODE SAMPLE PROMPTS (input file will be edited during peer evals)

## LLM SDK

## constrained decoding

## design decisions

## testing strategy

## current plan



## tasks for tomorrow
1. Create Pydantic models:
    ParameterDefinition
    FunctionDefinition
    PromptTest
    FunctionCallResult

2. Create JSON loader:
    load functions_definition.json
    load function_calling_tests.json
    handle missing files
    handle invalid json

3. Create dummy output writer:
    create fake FunctionCallResult
    write function_calling_results.json
    validate output shape