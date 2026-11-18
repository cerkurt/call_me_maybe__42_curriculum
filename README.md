*This project has been created as part of the 42 curriculum by cerkurt.*

# call_me_maybe

A local function-calling tool that translates natural-language requests into 
function names and typed arguments using a small language model, hybrid function selection, 
and token-level constrained argument extraction.

## Contents

- [Description](#description)
- [Instructions](#instructions)
- [Input and output formats](#input-and-output-formats)
- [Project structure](#project-structure)
- [Architecture diagram](#architecture-diagram)
- [Algorithm explanation](#algorithm-explanation)
- [Design decisions](#design-decisions)
- [Challenges encountered](#challenges-encountered)
- [Testing strategy](#testing-strategy)
- [Performance analysis](#performance-analysis)
- [Current limitations and remaining work](#current-limitations-and-remaining-work)
- [Resources and AI usage](#resources-and-ai-usage)

## Description

Given a request such as `What is the sum of 40 and 2?`, the program identifies
the appropriate function and extracts its input arguments. The result is a
structured call:

```json
[
    {
        "prompt": "What is the sum of 40 and 2?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 40.0,
            "b": 2.0
        }
    }
]
```

The output describes the call; the application does not execute the selected
function or calculate its return value.

Function definitions are supplied through JSON files. Their names,
descriptions, parameter names, and declared types drive the selection and
validation process. The currently supported parameter and return type
declarations are `number`, `string`, and `boolean`.

The supplied `Small_LLM_Model` SDK loads `Qwen/Qwen3-0.6B` by default.
The application uses its public encoding, decoding, and logit methods.
During argument extraction and regex option selection, invalid candidate
continuations are masked before the next token is selected. Function-name
selection instead scores complete available names and may rerank the two
leaders with a separate capability prompt.

## Instructions

### Requirements and installation

- Python **3.10+**, as specified in `pyproject.toml` and the project subject.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency and environment management.
- The supplied `llm_sdk/` workspace package.
- Internet access for the initial dependency and model downloads. Subsequent runs can reuse cached model files.

From the repository root, install the project dependencies:

```bash
make install
```

This target runs `uv sync`. Alternatively, the same installation can be performed directly:

```bash
uv sync
```

The application automatically creates missing output directories when writing results. 
Generated files under `data/output/` are excluded from Git and are not included in the 
submitted repository.

### Run the provided 11-prompt input

Run the application using its default input and output paths:

```bash
make run
```

The equivalent Python CLI command is:

```bash
uv run python -m src
```

The default paths are:

| CLI option | Default path |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

The standard CLI processes the provided prompts and writes a single JSON array containing 
their generated function calls. Each result includes the original prompt, selected 
function name, and extracted parameters.

The `make run` target also measures total execution time using `/usr/bin/time -p`. 
The reported `real` value represents the elapsed wall-clock time, including startup, model initialization, 
generation, and output writing.

If an error occurs during processing, the standard CLI stops the batch and does not 
write a new partial result file.

### Run one custom prompt

To test an individual request using the default function definitions, create a JSON file 
containing one prompt:

```bash
printf '[{"prompt":"What is the sum of 40 and 2?"}]\n' > /tmp/cmm_one_prompt.json
```

Run the application with the custom input path:

```bash
uv run python -m src \
  --input /tmp/cmm_one_prompt.json \
  --output data/output/one_prompt_result.json
```

Inspect the generated JSON:

```bash
uv run python -m json.tool data/output/one_prompt_result.json
```

The request text can be replaced with another prompt. The input file must remain a JSON 
array of objects containing the `prompt` field. Special characters and quotation marks 
must follow valid JSON escaping rules.

### Run with custom function definitions

The application reads available function names, descriptions, and parameter types from 
the supplied JSON definitions. It does not require hardcoded selection rules for individual 
function names.

Custom function definitions, prompts, and output paths can be supplied independently:

```bash
uv run python -m src \
  --functions_definition path/to/functions.json \
  --input path/to/prompts.json \
  --output data/output/custom_results.json
```

This allows the same application to process different sets of functions and natural-language 
requests without modifying the source code.

The standard CLI stops processing if a request raises an error. It does not automatically 
skip failed requests or evaluate generated calls against expected answers.

### CLI help and output inspection

Display the available command-line options:

```bash
uv run python -m src --help
```

Inspect the default output file:

```bash
uv run python -m json.tool data/output/function_calling_results.json
```

The JSON inspection command checks syntax and displays the generated structure. It does not 
verify whether the selected function or argument values match the user's intended request.

### Error handling

The application reports input, validation, generation, and output-writing errors at the CLI 
boundary. Handled failures terminate execution with a nonzero exit status and a descriptive 
error message.

These include missing input files, malformed JSON, invalid Pydantic models, unsupported 
parameter types, exhausted generation, and file-writing failures.

Input validation also rejects empty function definition lists, blank function names, and 
duplicate function names.

An empty prompt list is allowed and produces an empty result array. This is different from a list 
containing a prompt with an empty string, which represents a request entry with no text.

The application validates output parameter names and declared types before writing the results. 
Structural validation does not guarantee that the selected function and arguments are 
semantically correct.

### Code-quality checks

The included Makefile provides the required targets:

| Command | Description |
|---|---|
| `make install` | Install and synchronize project dependencies. |
| `make run` | Run the application and report execution time. |
| `make debug` | Run the application with Python's `pdb` debugger. |
| `make clean` | Remove Python bytecode and development cache directories. |
| `make lint` | Run Flake8 and mypy with the mandatory project flags. |
| `make lint-strict` | Run Flake8 and mypy with `--strict` (additional check). |

Run the required code-quality checks:

```bash
make lint
```

The `lint` target runs Flake8 and mypy with all flags required by the project subject:

- `--warn-return-any`
- `--warn-unused-ignores`
- `--ignore-missing-imports`
- `--disallow-untyped-defs`
- `--check-untyped-defs`

For additional type checking:

```bash
make lint-strict
```

The supplied SDK and configured development/test directories are excluded from direct lint checks. 
The application's production code remains subject to validation.

## Input and output formats

### Function definitions

The definitions file contains a JSON array. Each item describes one function:

```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {
                "type": "number"
            },
            "b": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    }
]
```

A declared return type describes the available function. It is not a result
the application needs to compute.

Pydantic converts these dictionaries into `FunctionDefinition` objects
containing nested `ParameterDefinition` and `ReturnDefinition` objects.

### Prompt tests

The prompts file contains an array of requests:

```json
[
    {
        "prompt": "What is the sum of 40 and 2?"
    }
]
```

Each item is validated as a `PromptTest`.

### Results

Each successfully processed request produces one FunctionCallResult containing 
the following output fields:

| Field | Meaning |
|---|---|
| `prompt` | The original request text. |
| `name` | The selected function name. |
| `parameters` | The complete dictionary of extracted argument values. |

### JSON value types

Parameter types are declared as `"number"`, `"string"`, or `"boolean"` in the 
function definitions. However, the generated output contains actual JSON values, 
not their type names.

For example, with appropriate custom function definitions, the output may look like this:

```json
[
    {
        "prompt": "Set the temperature to 18 degrees for room 'office'.",
        "name": "set_temperature",
        "parameters": {
            "room": "office",
            "degrees": 18.0
        }
    },
    {
        "prompt": "Disable notifications for room 'kitchen'.",
        "name": "set_notifications",
        "parameters": {
            "room": "kitchen",
            "enabled": false
        }
    }
]
```

In the output:

- `"office"` and `"kitchen"` are JSON strings.
- `18.0` is a JSON number, not a quoted string.
- `false` is a JSON boolean, not the string `"false"`.

The current numeric extractor converts recognized numeric values to Python `float` objects. 
JSON supports both integer and floating-point number representations for parameters 
declared as `"number"`.

## Project structure

The application separates command-line execution, data validation, model inference, 
and argument extraction into dedicated modules:

```text
.
├── src/
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # CLI, model initialization, error handling
│   ├── data_io.py               # JSON loading, validation, and writing
│   ├── data_models.py           # Pydantic schemas and parameter type checks
│   ├── generation_flow.py       # Main function-calling pipeline
│   ├── function_selectors.py    # Function ranking and capability reranking
│   ├── parameter_extractors.py  # Argument candidates and extraction prompts
│   ├── llm_helpers.py           # Token scoring and constrained decoding
│   └── regex_helpers.py         # Regex candidate selection and replacement
├── data/
│   ├── input/                   # Default function definitions and prompts
│   └── output/                  # Created at runtime; excluded from Git
├── tests/                       # Additional tests and evaluation utilities
├── llm_sdk/                     # Supplied uv workspace package
├── pyproject.toml
├── uv.lock
├── Makefile
├── .flake8
├── .gitignore
├── call_me_maybe_v1_5.pdf
└── README.md
```

The entry point, `src/__main__.py`, calls `generation_flow.progress_builder()`, which coordinates 
function selection, parameter extraction, regex handling, and final result validation.

The output directory is generated automatically when results are written and is not i
ncluded in the submitted repository.

## Architecture diagram

A detailed, interactive flowchart of the application is available on 
[FigJam](https://www.figma.com/board/3LcRyYZk3coNHkmvc3bKv7/call_me_maybe_codam_cerkurt?node-id=0-1&p=f&t=P96hPD09sWdrDlUF-0).

The diagram illustrates the complete execution flow, including input validation, hybrid 
function selection, parameter extraction, constrained decoding, regex handling, and result serialization.

A simplified Mermaid diagram is included in the following section to provide a 
high-level overview of the algorithm.

## Algorithm explanation

### End-to-end overview

```mermaid
flowchart TD
  A[Load JSON inputs] --> B[Validate with Pydantic]
  B --> C[Initialize one SDK model]
  C --> D[For each user request]
  D --> E[Hybrid function selection]
  E --> F[Find selected FunctionDefinition]
  F --> G[Extract parameters using constrained decoding]
  G --> H[Optional regex post-processing]
  H --> I[Validate parameters and create FunctionCallResult]
  I --> J{More requests?}
  J -->|Yes| D
  J -->|No| K[Write one JSON array]
```

`progress_builder()` orchestrates the requests. It does not execute
functions; it produces descriptions of calls. It reuses the same
`Small_LLM_Model` instance across requests.

### 1. Select the function: full scoring inside the hybrid selector

**Full scoring and hybrid selection are connected stages of one selector,
not two independent modes selected by a CLI flag.** For every request,
`generation_flow.progress_builder()` calls
`function_selectors.hybrid_function_selector()`. That function first ranks
every available function by a full-sequence score. Only a close result
triggers the additional capability comparison.

`build_function_selector_prompt()` includes the actual request and the
supplied function names, descriptions, and parameter names/types. System
and user chat messages are followed by the open assistant prefix:

```text
<|im_start|>assistant
<think>

</think>

```

#### A. Full scoring: compare each complete function name

`llm_helpers.score_function_candidate()` evaluates a **known continuation**.
It encodes one function name and scores all its tokens in order. At each
step, it obtains logits, converts the specified token's score into a log
probability, then appends that specified token to the context to score
the next one. It does not select the highest-logit token at that step.
The prompt context is reset before evaluating the next candidate name.

```text
name_score = sum of log P(token_i | prompt and earlier name tokens)
```

The code computes the normalization with a stable log-sum-exp calculation:
It subtracts the largest logit before exponentiation. The final score is
a sum of conditional log-probabilities, not a percentage. A larger score
is better; for example, -1.2 ranks above -4.6. Scores are not divided by
the number of tokens, so candidate-name length can affect ranking.

Why compare complete names? A greedy token-by-token decoder commits to the 
strongest allowed next token without evaluating the full continuation of 
every candidate. A locally preferred token does not necessarily begin 
the strongest complete continuation. Consider these **invented** scores 
after a shared prefix, with two tokens left in each candidate:

| Candidate | First differing token | Following token | Total |
|---|---:|---:|---:|
| A | -0.6 | -4.0 | -4.6 |
| B | -1.0 | -0.2 | -1.2 |

Greedy selection would take A's first differing token because -0.6 is
higher than -1.0. Full scoring evaluates both complete continuations and
prefers B. This gives every supplied name a complete evaluation without
allowing an invented name. It is a different selection criterion, not a
guarantee that the model's preferred name has the right meaning.

#### B. Hybrid selection: use an extra check when the leaders are close

The selector retains the best and second-best full scores and calculates:

```text
margin = best_score - second_score
```

```mermaid
flowchart TD
    A[Score every complete available function name] --> B{Second candidate exists?}
    B -->|No| C[Return the only scored candidate]
    B -->|Yes| D{Score margin at least 2.0?}
    D -->|Yes| E[Return the best full-score candidate]
    D -->|No| F[Capability check for the top two candidates]
    F --> G[Compare Yes score minus No score]
    G --> H[Return the higher capability score]
```

A gap of at least **2.0** returns the full-score leader directly. A smaller
gap runs `capability_score()` for each of the top two definitions. This
helper prepares a separate prompt asking whether that one function can
fulfill the request using suitable arguments. It includes the function's
description and parameter declarations. The same SDK model then scores
the fixed answers `Yes` and `No`:

```text
capability = full_score(Yes) - full_score(No)
```

The code scores **both** answers; it does not need to freely generate a
Yes/No explanation. The larger difference wins. If capability scores
tie, the original full-score leader wins. The original leader is also
returned when it is the only candidate after scoring.

The `2.0` threshold is a heuristic applied to the difference between accumulated
log-probability scores. It is not a calibrated confidence percentage. A larger
threshold would trigger capability reranking more often, potentially increasing
execution time without necessarily improving accuracy.

For an **illustrative** example, full scores -2.1 and -2.8 have a margin
of 0.7, which triggers the extra check. If the capability differences
are 0.2 and 3.0 respectively, the second candidate replaces the original
leader. With scores -1.0 and -4.0, the margin is 3.0 and no extra check runs.
These numbers explain control flow; they are not measured test outputs.

Capability reranking considers only the two highest-ranked functions from the first
stage. Therefore, it cannot recover the correct function if that function was
ranked outside the top two.

#### Why the extra stage was introduced and what it costs

The additional question separates two judgments: how likely the function's
name is as a continuation of the selection prompt, and whether that
function's described capability fits the request. Requests involving
negation or overlapping capabilities exposed wrong name selections during
development. A capability prompt gives close candidates another comparison
based on one definition at a time rather than the entire displayed list.
It uses the same model, not a second model or an independent correctness
oracle.

The margin rule limits the cost of this second comparison. Close requests
require four extra continuation scores: Yes and No for each of two
functions. Each score can require multiple SDK logit calls. All function
names are already scored before the margin check, so this approach does
more work than one greedy name continuation.

This design provides a complete-name comparison and a bounded opportunity to 
revise an ambiguous ranking, at the cost of additional inference.

The `2.0` threshold is a heuristic based on the difference between accumulated 
log-probability scores, not a calibrated confidence measure. A large margin can 
skip a potentially useful capability check. Furthermore, reranking is limited to 
the top two candidates, so it cannot recover a correct function ranked lower. 

Increasing the threshold would trigger more reranking but would not eliminate 
this shortlist limitation.

#### Relation to constrained argument decoding

Function selection returns a name from the supplied definitions by scoring
fixed names and, when needed, capability answers. After that, argument
values and regex option codes use `generate_name_from_candidates()` with
token-prefix constraints and logit masking.

### 2. Determine parameter extraction strategy

After matching the selected name back to its `FunctionDefinition`, the
pipeline recognizes a regex-specialized schema **once per selected
function** and creates a new parameter dictionary.

For every declared parameter, it either skips the special `regex` field
(to be generated later), or selects an extraction strategy by its type:

| Declared type | Candidate construction |
|---|---|
| `number` | Extract signed integers, decimals (including `.5`), scientific notation (`1e3`) and comma-grouped forms (`1,000`), then convert to finite floats. |
| `string` | Collect quoted contents, individual chunks, two- and three-word segments outside recognized quotes, and chunks with trailing .!? removed. Deduplicate in first-seen order. Do not append the full request as a fallback. |
| `boolean` | Offer the fixed candidate set `[True, False]`. |

If a number parameter has **no numerical candidates**, a descriptive
`ValueError` is raised rather than fabricating a value. For strings,
both candidate coverage and the model's semantic selection remain
possible sources of error.

`message_for_parameter_extraction()` constructs the LLM prompt using:
the original request, selected function and description, parameter
names/types, **the current parameter**, and values already extracted.
Earlier values help provide context for subsequent parameters, but do
not prevent the same valid value from being used more than once.

### 3. Prepare candidates and run constrained token decoding

Each candidate is serialized using `json.dumps()` and encoded into a
token-ID sequence with the SDK's `encode()` method. The implementation
prepends a space before the serialized candidate, matching the open JSON
prefix of the parameter prompt. Tokenization is sensitive to whitespace
and position: `encode(prefix) + encode(value)` is not necessarily the
same token sequence as `encode(prefix + value)`.

`generate_name_from_candidates()` uses the token sequences to constrain
greedy selection, rather than random sampling. At every generation step:

1. `find_allowed_next_token_ids()` identifies the next token IDs for
   every candidate still matching the generated prefix.
2. `get_logits_from_input_ids()` obtains next-token scores from the
   model, conditioned on the prompt plus previously generated tokens.
3. `select_next_token_id()` masks every disallowed logit to `-inf`,
   without changing token indexes, and selects the strongest allowed ID.
4. The generated ID extends both the full model context and the
   candidate-matching prefix. Generation stops when a complete candidate is
   reached. A `ValueError` is raised if no valid continuation is available
   or the generation token limit is exhausted.

For illustration, imagine that two candidate values are encoded as:

```text
candidate_a → [10, 20]
candidate_b → [10, 30]
```

| Already generated | Allowed next tokens |
|---|---|
| `[]` | `{10}` |
| `[10]` | `{20, 30}` |
| `[10, 20]` | Completed `candidate_a` |

The selected candidate is returned as JSON text and converted to the
corresponding Python value with `json.loads()`. It is stored under the
current parameter name in `parameter_dict`. The cycle repeats until all
ordinary parameters have been visited.

A constrained output is **guaranteed to be selected from the supplied
candidates when generation succeeds**, but it is not guaranteed to be
the semantically correct choice.

### 4. Complete regex parameters when recognized

A function activates regex-specific processing only when its definition
contains `source_string`, `regex`, and `replacement`, **all declared as
strings**. Detection depends on the parameter schema, not the function
name.

The `regex` field is skipped during the ordinary extraction loop; the
other parameters are extracted first. After the loop:

1. `convert_regex_replacement_value()` normalizes supported unquoted character 
   punctuation — plus eligible request-derived quoted literals.
2. `regex_candidate_generator()` provides seven category patterns —
   vowels, letters, digit sequences, uppercase, lowercase, whitespace,
   punctuation — plus eligible request-derived quoted literals.
3. `generate_regex_pattern()` maps these to short option labels and
   uses constrained token selection to choose the best one.
4. The selected option is translated back to a regex pattern and stored
   in `parameter_dict["regex"]`.

Examples include `[aeiouAEIOU]` for vowels and `[0-9]+` for digit
sequences. This is **bounded pattern selection**, not arbitrary regex
synthesis. The application does not call `re.sub()` as part of function
execution; replacement examples were verified separately in tests.

### 5. Validate and write the structured result

`parameter_type_checker()` confirms that all required parameter names are
present, no unexpected names were added, and values match their declared
types. The numeric check accepts `int` values and finite `float` values but rejects
`bool`, which Python otherwise considers a subclass of `int`.

Successful requests yield one `FunctionCallResult` each, with precisely
`prompt`, `name`, and `parameters`. After the request loop completes,
`model_dump()` provides Python dictionaries, and
`json.dumps(..., allow_nan=False)` serializes the array to UTF-8 JSON.

Successful requests yield one `FunctionCallResult` each, containing `prompt`, 
`name`, and `parameters`. After the request loop completes, `model_dump()` converts 
the result models into Python dictionaries, and `json.dumps(..., allow_nan=False)` 
serializes the complete list. The output writer creates missing parent directories and 
writes the resulting JSON text using UTF-8 encoding.

Pydantic and the final parameter checker validate structure and types. They cannot prove 
that a chosen function or argument corresponds to the user's intended meaning. 
This distinction is evaluated separately in the testing strategy.

## Design decisions

- **Separate I/O, data models, generation, and orchestration.** Each module
  has a distinct responsibility. Reusable functions receive file paths, validated
  objects, or an existing SDK model rather than managing unrelated tasks.

- **Reuse one model instance.** The SDK model is initialized once, after input
  validation, and shared across requests to avoid repeated loading costs.

- **Use dynamic function definitions.** Function names, descriptions, and parameter
  types are read from the supplied JSON file instead of hardcoding rules for individual
  functions. This allows the same selection pipeline to work with new function
  definitions without modifying the source code.

- **Combine full-sequence scoring with conditional capability reranking.** Scoring
  complete function names avoids committing to an early greedy token choice. Capability
  reranking is reserved for close candidates to balance selection quality and inference cost.

- **Constrain argument generation to predefined candidates.** Prompt wording provides
  semantic context, while token-level logit masking restricts argument values and regex
  option codes to available candidates.

- **Retain extraction context.** Previously extracted parameter values are included in the
  prompt for subsequent parameters, helping the model distinguish between arguments without
  preventing valid repeated values.

- **Share prompt assembly across parameter types.** A single prompt builder handles number,
  string, and boolean parameters, using type-specific instructions while sharing function
  metadata, extracted values, chat roles, and JSON formatting.

- **Expand the candidate pool before constraining selection.** Quoted strings and word n-grams
  make multiword arguments available without assuming that every quoted phrase is the intended
  value. Constrained decoding then selects from the constructed pool.

- **Defer regex pattern extraction.** Recognized regex schemas extract the source string and
  replacement first, allowing a specialized, bounded pattern-selection step to use those values
  as context.

- **Accumulate one complete result per request.** Parameter dictionaries are initialized
  outside the inner parameter loop, so all extracted arguments are collected before constructing
  a single `FunctionCallResult`.

- **Report errors at the CLI boundary.** Lower-level functions propagate errors, while the CLI
  handles them with descriptive messages and nonzero exit statuses.

## Challenges encountered

### Whitespace changes token boundaries and model scores

A leading or trailing space is part of the token sequence. During development,
moving a space between the JSON prefix and the value candidate changed the
IDs presented to the model and the allowed next-token choices.

For example, the following were measured through the SDK tokenizer:

| Text supplied to `encode()` | IDs |
|---|---|
| A single space | `[220]` |
| A double quote | `[1]` |
| A space followed by a double quote | `[330]` |
| `"cat"` | `[1, 4616, 1]` |
| A space followed by `"cat"` | `[330, 4616, 1]` |
| `"cat"` followed by a space | `[1, 4616, 1, 220]` |
| A minus sign | `[12]` |
| A space followed by a minus sign | `[481]` |

These IDs describe the inspected tokenizer, not a universal vocabulary.

Consider two ways to construct the same visible boundary:

```text
A: encode(prefix ending in ": ") + encode("\"cat\"")
B: encode(prefix ending in ":")  + encode(" \"cat\"")
```

The concatenated text can look identical while the concatenated token-ID
lists differ. In general:

```text
encode(prefix) + encode(value) != encode(prefix + value)
```

In case A, a separate space and quote can produce IDs `220` and `1`.
In case B, the combined space-plus-quote token can be `330`.
The model is then evaluating different token continuations at different
positions, so the logits are not directly interchangeable.

Whitespace also has a data-level meaning:

| JSON text | Decoded value |
|---|---|
| `"cat"` | Three characters: `cat` |
| `"cat" ` | The same three-character value; the space is outside the string. |
| `"cat "` | Four characters; the final space belongs to the argument. |

The current implementation ends the JSON prefix at the colon and prepends a 
space to each serialized candidate. This keeps the tokenization convention 
consistent during constrained decoding. Whitespace placement affects token 
boundaries but does not guarantee semantically correct selection.

### A correct value must be present in the candidate pool

One request exposed a candidate-extraction error:

```text
Set the temperature to 18 degrees for room 'office'.
```

The outside-quotes splitter retained `'office'.`. Its final character was
a period, so the condition that removed matching outer quotes did not apply.
The plain `office` value was absent from the pool, making it unavailable
to constrained selection.

The inside-quotes helper already found `office`. Its results were added
to the candidate pool for multi-parameter functions, which fixed the
temperature example.

This led to a useful debugging order: inspect candidate construction before
changing the model prompt.

### Multiword values needed both a larger pool and clearer instructions

An earlier pool could offer only separate words for an unquoted full name.
For `Greet Jane Doe`, selecting either `Jane` or `Doe` loses part of
the requested value. Adding contiguous two- and three-word phrases makes
`Jane Doe` available without manually encoding a specific person's name.

Pool construction and prompting solve different parts of this problem.
The shared builder's string instruction asks for the complete current
argument and distinguishes it from the command. In
`Greet Jane Doe using the word 'hello'`, both the name and the quoted
word remain candidates; the function description and parameter name must
guide the model to the name.

The current implementation deduplicates words, phrases, quoted spans,
and punctuation-trimmed chunks. Unlike that earlier version, it does not
append the entire request as a fallback. It does not enumerate every possible
substring. A targeted pool check still finds no
`First Second Third Fourth` candidate for the longer command
`Greet First Second Third Fourth`.

### Valid JSON regexes could still match the wrong characters

The experimental free-string path could produce a complete JSON string
whose content was the wrong regex. For example, `[a-zA-Z]+` is a valid
pattern but replaces whole letter sequences when the request asks only
for vowels. Testing only `json.loads()` would miss that error.

Extended checks therefore applied the selected pattern to source text:

| Source | Pattern | Replacement | Expected substitution result |
|---|---|---|---|
| `Room 12` | `[0-9]+` | `X` | `Room X` |
| `Programming is fun` | `[aeiouAEIOU]` | `*` | `Pr*gr*mm*ng *s f*n` |
| `The cat sat on the mat with another cat` | `cat` | `dog` | `The dog sat on the mat with another dog` |

Selecting short option codes from a described pattern catalog gave a
usable bounded path. The code, descriptive label, regex text, and encoded
token sequence have separate roles. Encoding the option codes constrains
the model's answer to the displayed choices; mapping back supplies the
actual pattern. Prompt wording still affects which allowed option wins.
The catalog is therefore documented as limited support, rather than a
guarantee of arbitrary regex generation.

The literal replacement `asterisks` exposed a separate issue: even the
correct vowel pattern inserts that word repeatedly unless the requested
symbol is interpreted as `*`. The small quote-aware character-name helper
addresses the supported cases while preserving quoted literal text.

### Numeric parsing improvements do not guarantee the chosen argument

The initial numeric pattern incorrectly split values such as `.5`, scientific 
notation, and comma-grouped numbers. The updated extractor recognizes these 
formats and converts `.5` to `0.5`, `1e3` to `1000.0`, and `1,000` to `1000.0`.

The extractor also preserves signed values, including both `0.5` and `-0.5` 
in requests such as `Add 0.5 and -0.5.`. Previously selected values are not 
removed from subsequent candidate pools, allowing valid repeated arguments 
such as `Add 7 and 7.`.

Numeric extraction and semantic selection remain separate responsibilities: 
making a value available does not guarantee that the model will assign it to 
the correct parameter.

### Quoted text, escape characters, and Unicode require careful handling

Natural-language requests may contain quoted strings, apostrophes, escaped 
characters, and Unicode text. These must be handled carefully to preserve the 
intended argument value.

For example, `Reverse 'hello world' please` contains a quoted argument with an 
internal space. Splitting the request only on whitespace would lose that grouping.

String processing also involves several interpretation layers:

1. **Python strings:** Escape sequences may be interpreted before a function receives a string.
2. **JSON input:** Strings must follow JSON escaping rules, including escaped quotation marks and control characters.
3. **Natural-language requests:** Quotes may delimit an argument or be part of its content.
4. **JSON output:** Extracted values must be serialized without changing their intended meaning.

For example, a newline character and the literal characters `\` and `n` are 
different Python strings:

```python
actual_newline = "\n"
escaped_newline = "\\n"

assert len(actual_newline) == 1
assert len(escaped_newline) == 2
```

Unicode characters introduce another representation distinction. For example, 
the JSON string:

```json
{
    "text": "\u00c7a\u011fr\u0131 \ud83c\udf21\ufe0f"
}
```

decodes to the same text as:

```text
Çağrı 🌡️
```

The output writer uses UTF-8 encoding and `ensure_ascii=False` so Unicode characters 
remain readable in the generated JSON file. Escaped and literal Unicode representations 
may look different in the file while decoding to the same string value.

During development, inspecting `repr()` and testing JSON round trips with `json.loads()` 
helped distinguish actual string contents from their serialized representations.

The current string extractors handle the tested quoting cases, but escaped quote delimiters 
and ambiguous natural-language quoting remain areas where broader coverage is needed.

### Schema validity and the intended meaning need different checks

A generated argument can have the correct JSON type while still representing 
the wrong value. For example, selecting `"string"` instead of `"hello"` for 
`Reverse the string 'hello'` produces a valid string argument but does not 
satisfy the request.

Similarly, `disabled: true` is structurally valid for a boolean parameter but 
contradicts an instruction to enable notifications.

Constrained decoding guarantees candidate membership when generation succeeds, 
while Pydantic and the parameter checker validate structure and types. Semantic 
correctness requires a separate comparison against the intended function call.

### Function selection was sensitive to prompt structure

During development, adding another function definition exposed incorrect 
function-name selections. The model's ranking changed significantly depending 
on how function metadata and the user request were formatted.

Using explicit system and user message boundaries, followed by the model's 
expected open assistant prefix, improved selection behavior in targeted diagnostics.

This demonstrated that model scores depend not only on the candidate function 
names but also on the surrounding prompt context. The final hybrid selector retains 
structured prompting while evaluating complete function-name continuations and 
conditionally reranking close candidates.

### Exceptions belong at the right boundary

A missing file already raises an error from `open()`. A malformed JSON
document raises a decoding error from `json.load()`. A mismatched Pydantic
field raises a validation error.

A lower-level `except` block that only re-raises the same error adds no
new handling. The CLI provides context and an exit status for handled
failures. Stopping after a failed generation also avoids continuing with
a result variable that was never assigned.

Result serialization is completed before output directories are created or 
files are opened, preventing serialization failures from modifying existing 
output files.

## Testing strategy

Testing checks input loading and Pydantic validation, function-name selection,
number/string/boolean candidate generation, constrained token decoding, semantic
argument accuracy, output schema, edge cases, failure reporting, and end-to-end
CLI behavior. Ground-truth function calls were specified separately from model output.

### Recorded outcomes (November 16–17, 2026)

These are recorded development runs, not fresh inference benchmarks from
this documentation review. The saved default output was checked against
all 11 expected calls. The local combined report was independently
recounted: its actual-versus-expected records confirm 71/76 complete calls,
73/76 correct names, and one error. The full 31-case report is not present
in the current workspace; its 19/31 figure and six errors remain a
historically reported result rather than a newly audited measurement.

| Test set | Available functions | Prompts | Fully correct | Accuracy | What this measures |
|---|---:|---:|---:|---:|---|
| Supplied/default 11-prompt input | Original supplied functions | 11 | 11 | **100%** | Standard CLI, original assignment input |
| Original extended regression set | 8 | 55 | 54 | **98.2%** | Additional prompts and edge cases |
| Independent generalization set | 7 new functions | 31 | 19 | **61.3%** | Unseen function names, overlapping capabilities, harder boolean/string cases |
| Curated combined regression set | 15 | 76 | 71 | **93.4%** | All prior 55 plus 21 selected generalization examples |

The 76-case run reported **73/76 correct function names** and **one execution
error**. The original 55 cases retained **54/55** correct results after adding
the new competing definitions. The 31-prompt independent set used a separate
seven-function definition list; six prompts raised errors and were counted as
failures, not omitted from its 61.3% metric.

**Important:** These are different, manually designed datasets, not directly
comparable estimates of accuracy on unknown evaluation inputs. The combined
set is curated and includes many cases already known to succeed. Its 93.4%
accuracy does not supersede the independent generalization result of 61.3%.

### Failure analysis from the combined 76-case suite

| Case | Observed behavior | Classification |
|---|---|---|
| `OLD-34` | `Greet the person by name and surname: Jane Doe` returned only `Jane` | Incomplete multiword string extraction |
| `NEW-10` | Reminder request could not obtain required `hour` for an incorrectly selected alarm function | Function selection and subsequent missing-candidate error |
| `NEW-16`, `NEW-17` | Discount requests selected regex substitution instead | Candidate-set sensitivity / incorrect function selection |
| `NEW-28` | `unmute it` yielded `muted: true` instead of `false` | Boolean semantic interpretation |

The reminder issue was investigated with full-sequence scores. For `NEW-10`,
`configure_alarm` ranked first at **-0.8778**, while `create_reminder` ranked
fifth at **-10.2577**. The margin between the top two was **2.2146**, above
the **2.0** reranking threshold. Independent capability scoring favored
`create_reminder` (**6.2656**) over `configure_alarm` (**4.0938**), but the
correct function was outside the top-two shortlist and capability reranking
did not run in the normal hybrid path.

In the separate 31-case generalization run, problems also appeared with
apostrophes inside quoted strings, empty-string selection, boolean negation,
and overlapping capabilities (`reminder` versus `alarm`).

Two discount cases passed with seven available functions but selected the wrong
function when the combined list expanded to fifteen. This demonstrates that
adding competing definitions can change semantic function selection even when
the output remains structurally valid.

### Reproducing the mandatory/default test run

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
uv run python -m json.tool data/output/function_calling_results.json
```

### Inspecting stored diagnostics and running local fixtures

The local combined report can be inspected without running the model:

```bash
uv run python -m json.tool ../combined_evaluation/combined_run_report.json
```

It records each expected call, observed call or error, and pass flag. This
report is a development artifact outside the repository. The accompanying
runner requires the path and module updates described in Instructions
before it can reproduce the report with the current project layout.

The independent 31-request input can be passed to the standard CLI:

```bash
uv run python -m src \
  --functions_definition ../test_inputs/generalization_functions.json \
  --input ../test_inputs/generalization_prompts.json \
  --output data/output/generalization_results.json
```

This exercises normal fail-fast batch behavior; it is not a command for
reproducing a per-request report that continues after six errors. The old
local `test_generalization.py` also refers to moved input files and a
missing expected-results file, so it needs maintenance before reuse.

### Validation and negative input tests

Explicit bad-input checks passed for malformed JSON, missing required fields,
empty function lists, and duplicate function names. An unsupported parameter
type is rejected by Pydantic. Tests also covered missing numeric argument
candidates, Unicode, quoted/empty strings, scientific notation, negative
numbers, thousands separators, multiple parameters, regex substitution, and
boolean instructions with negation.

For example, deliberately pass an invalid JSON file:

```bash
printf '[{"name":"broken",}]\n' > /tmp/cmm_bad.json
uv run python -m src --functions_definition /tmp/cmm_bad.json
```

This should report a JSON parse error and exit nonzero without starting the model.

## Performance analysis

Previously recorded cached-model timings on the development machine:

The figures below are retained from the author's run record. Raw timing
logs are not bundled in this checkout, and this documentation review did
not run model inference again to remeasure them.

| Workload | End-to-end runtime | Result |
|---|---:|---|
| Default 11-prompt CLI | **46.617 s** | 11/11 fully correct |
| Original 55-prompt CLI | **4 min 13.26 s** | 54/55 fully correct |

The default and 55-prompt measurements were obtained using the real
`Small_LLM_Model` SDK and the regular batch CLI. The combined 76-prompt
regression set was run using a shared-model per-prompt diagnostic runner;
a comparable total runtime for that run was not recorded. Per-prompt
subprocess tests (which reload the model repeatedly) are not a valid
measurement of a single-batch run.

The subject requests 90%+ accuracy, 100% JSON/schema compliance for outputs,
and processing of all evaluation prompts in under five minutes on standard
hardware. The 55-prompt run met the five-minute target on the development
machine, but this does not guarantee the same runtime with different models,
function counts, prompts, or evaluator hardware.

For another local timing measurement:

```bash
time uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The model is initialized once per batch. Repeated SDK logit calculations,
context length, candidate pool size, and token limits affect runtime.
There is a clear trade-off between fuller comparison of alternative
functions and inference time.

## Current limitations and remaining work

1. **Function selection is sensitive to the candidate set.** Full-sequence
   scoring followed by top-two capability reranking can miss a correct
   function ranked outside the shortlist. The reminder/alarm example in
   `NEW-10` demonstrates this. Adding competing functions can also change
   results: `NEW-16` and `NEW-17` passed with seven functions but chose
   regex substitution in the fifteen-function run.
2. **Total log-probability is length-sensitive.** Scores are summed across
   name tokens, which may favor shorter candidate token sequences. This
   may contribute to some rankings, but the specific failures above were
   not proven to be caused by name length.
3. **String candidate coverage and selection are imperfect.** The pool
   supports quoted spans, individual chunks, two- and three-word n-grams,
   and punctuation-trimmed chunks. It does not append the full request
   as a fallback. The correct value can still lose to an incomplete
   candidate (`OLD-34`: `Jane` instead of `Jane Doe`). Ambiguous
   apostrophes, escaped delimiters, and some empty-string instructions
   also remain challenging.
4. **Boolean polarity is not always inferred correctly.** Negated or
   inverse-state commands (`unmute` with `muted`) may yield an incorrect
   boolean even though both `True` and `False` are valid candidates.
5. **Numeric extraction and selection are separate.** Signed, decimal,
   scientific, and comma-grouped numeric forms are supported, but
   written-out numbers and locale-specific representations are not
   generally converted. Missing numerical candidates raise `ValueError`
   instead of inventing a number.
6. **Regex support is deliberately bounded.** Only a recognized schema
   containing string `source_string`, `regex`, and `replacement` activates
   post-processing. Supported patterns come from a small category catalog
   plus certain quoted literals; arbitrary regex synthesis is not provided.
7. **Valid structure does not imply correct meaning.** Pydantic and the
   result checker verify names/types relative to the selected definition,
   but a structurally valid call can select the wrong function or value.
8. **Batch CLI behavior is fail-fast.** A missing required argument can
   stop the normal batch. The optional test runner catches errors per
   request solely to measure the remainder of a regression set.
9. **Final submission verification remains necessary.** The November 17
   review confirmed clean flake8 and the mandatory mypy flags, complete
   module/function/class docstrings, and the required Makefile targets.
   A fresh-clone run still needs its final check. At review time,
   `data/output/function_calling_results.json` was still tracked despite
   the ignore rule; remove generated output from Git's tracked submission.
10. **Optional artifacts are not fully portable yet.** The FigJam PDF/PNG
   exports are absent, while diagnostic fixtures live above the repository
   and their old runners require path/import updates. Default CLI usage
   does not depend on these optional artifacts.

These limitations are documented from observed tests; the reported percentages
are not guarantees for unknown evaluation prompts.

## Resources and AI usage

### References

- [Project subject v1.5](call_me_maybe_v1_5.pdf): mandatory behavior, output
  format, evaluation targets, and README requirements.
- [Supplied SDK source](llm_sdk/llm_sdk/__init__.py): public methods and model
  initialization.
- [Python JSON documentation](https://docs.python.org/3/library/json.html):
  loading, serialization, escaping, and encoder options.
- [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259): JSON syntax, numbers, and
  string representations.
- [Python regular expressions](https://docs.python.org/3/library/re.html):
  candidate-number extraction and escape syntax.
- [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/):
  nested models, validation, and serialization.
- [uv project guide](https://docs.astral.sh/uv/guides/projects/) and
  [additional run dependencies](https://docs.astral.sh/uv/concepts/projects/run/#requesting-additional-dependencies):
  workspace environments and command execution.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B):
  model information and chat behavior.
- [Hugging Face chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating):
  message roles and generation prefixes, consulted for understanding the
  prompt format. The application accesses the model through the supplied SDK.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762):
  background on the Transformer architecture.

### How AI was used

ChatGPT/Codex was used as a **learning, debugging, and documentation partner**
in these areas:

- **Concepts and SDK exploration:** Python collections, Pydantic models,
  logit masking, tokenization boundaries, JSON encoding/decoding, and the
  supplied SDK's public methods.
- **Algorithm design and review:** Discussing finite candidate pools,
  full-sequence log-probability scoring, capability-based reranking,
  sequential argument extraction, and the regex special case.
- **Tests and debugging:** Designing normal and edge-case prompts,
  generating independent expected results, analyzing wrong function
  choices and argument values.
- **Documentation:** Reviewing README completeness against the project
  subject, preparing evaluation-friendly descriptions of trade-offs and
  known limitations.
