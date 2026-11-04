"""Manual SDK experiments, selected by the function call at the bottom.

Run from the project root with: uv run python -m sdk_playground.
The active experiment prints selected names and their parameter schemas.
"""

from llm_sdk import Small_LLM_Model
from src import data_io, data_models, generators

import json
import datetime


def llm_model_test() -> None:
    """Explore tensor shapes, vocabulary, scores, decoding, and generation."""
    model_object = Small_LLM_Model()

    test_str = "Hello LLM"
    # encode() produces a batch with shape (1, token_count).
    test_result = model_object.encode(test_str)

    print("Encoded text result")
    print(type(test_result))
    print(test_result)
    print(test_result.shape)
    print()

    # [0] selects the one sequence to decode as text.
    reverse_test_str = model_object.decode(test_result[0])

    print("Decoded text result (reversed from encoding)")
    print(type(reverse_test_str))
    print(reverse_test_str)
    print()

    file_path = model_object.get_path_to_vocab_file()

    print("Vocab file path")
    print(type(file_path))
    print(file_path)
    print()

    print("Printing vocab file content")
    # Vocabulary JSON is a dictionary; the generic loader expects an array.
    with open(file_path, "r") as json_file:
        file_content = json.load(json_file)
        print(len(file_content))
        print()

    print("Finding the exact token we want")
    for token, token_id in file_content.items():
        if token_id == 9707:
            print(token)
            print()

    print("Content of the ID list")
    token_ids_list = test_result[0].tolist()
    print(type(token_ids_list))
    print(token_ids_list)
    print()

    print("Get logits")
    # A score's position is a token ID for the next token in the context.
    logit_list = model_object.get_logits_from_input_ids(token_ids_list)
    print(type(logit_list))
    print(len(logit_list))
    print()

    print("Printing the ID from the max logit")
    next_token_id = logit_list.index(max(logit_list))
    print(max(logit_list))
    print(next_token_id)
    print()

    print("Printing the decoded text of the next token")
    selected_token_ids = []
    selected_token_ids.append(next_token_id)
    print(model_object.decode(selected_token_ids))
    print()

    token_ids_list.append(next_token_id)
    print(model_object.decode(token_ids_list))
    print()

    print("Get new logits")
    logit_list = model_object.get_logits_from_input_ids(token_ids_list)
    print(type(logit_list))
    print(len(logit_list))
    print()

    print("Printing the new ID from the max logit")
    next_token_id = logit_list.index(max(logit_list))
    print(max(logit_list))
    print(next_token_id)
    print()

    print("Printing new the decoded text of the next token")
    selected_token_ids = []
    selected_token_ids.append(next_token_id)
    print(model_object.decode(selected_token_ids))
    print()

    token_ids_list.append(next_token_id)
    print(model_object.decode(token_ids_list))
    print()

    print("Creating a loop for continous tokens")
    test_model = Small_LLM_Model()
    test_str = "Hello, my name is Ceren. I am a student at a coding school."
    encode_result = test_model.encode(test_str)
    all_token_ids = encode_result[0].tolist()

    generated_token_ids = []

    for _ in range(5):
        next_token_logits = test_model.get_logits_from_input_ids(all_token_ids)
        next_token_id = next_token_logits.index(max(next_token_logits))
        generated_token_ids.append(next_token_id)
        all_token_ids.append(next_token_id)

    print(test_model.decode(generated_token_ids))
    print(test_model.decode(all_token_ids))


def constrained_decoding_test() -> None:
    """Print the highest-scoring allowed IDs using a small invented mapping.

    These IDs belong to this example's token_ids dictionary. They are
    independent of the real SDK vocabulary. Tied winners are all printed.
    """
    token_logits = {"maybe": 9.0, "yes": 7.0, "no": 5.0}
    token_ids = {"maybe": 0, "yes": 1, "no": 2}
    allowed_ids = [1, 2]

    allowed_results: dict[int, float] = {}
    for token_name, token_logit in token_logits.items():
        token_id = token_ids[token_name]
        if token_id in allowed_ids:
            allowed_results[token_id] = token_logit

    max_score = max(allowed_results.values())
    for token_id, token_logit in allowed_results.items():
        if token_logit == max_score:
            print(token_id)


def generator_test() -> None:
    """Generate two continuations while reusing one loaded model."""
    test_object = Small_LLM_Model()
    result1 = generators.generate_text(test_object, "this is my start text", 5)

    result2 = generators.generate_text(
        test_object, "this is my start text", 10
    )

    print(result1)
    print(result2)


def prompt_builder_test() -> None:
    """Display selection instructions built from validated definitions."""
    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )
    test_result = generators.build_selection_prompt(
        "Greet shrek", validated_functions
    )
    print(test_result)


def object_reading_test() -> None:
    """Inspect unrestricted continuation of a function-selection prompt."""
    model_object = Small_LLM_Model()
    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )
    promp_selection = generators.build_selection_prompt(
        "Greet shrek", validated_functions
    )

    test_result = generators.generate_text(model_object, promp_selection, 20)

    print(test_result)


def function_name_encoding_test(model_object: Small_LLM_Model) -> dict:
    """Return function-name to token-ID-list pairs using the supplied model.

    Use this same model for generation so candidate IDs and generated
    IDs come from the same tokenizer.
    """
    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )

    name_id_match: dict = {}
    for function in validated_functions:
        encoded_name = model_object.encode(function.name)
        function_name = function.name
        token_id_list: list = encoded_name[0].tolist()
        # token_count = len(token_id_list)
        name_id_match.update({function_name: token_id_list})

    #     print(
    #         f"\nFunction name: {function_name}\n"
    #         f"Token IDs: {token_id_list}\n"
    #         f"Token count: {token_count}"
    #     )

    # print(json.dumps(name_id_match, indent=4))
    return name_id_match


def id_finder_test() -> None:
    """Inspect allowed next IDs for one manually specified name prefix."""
    model_object = Small_LLM_Model()
    test_dict = function_name_encoding_test(model_object)
    result_list = generators.find_allowed_next_token_ids(
        test_dict, [8822, 3062, 39794, 12993]
    )

    print(result_list)


def token_id_selector_test() -> None:
    """Trigger the selector's ValueError for an empty allowed-ID list."""
    result = generators.select_next_token_id([9.0, 7.0, 5.0], [])

    print(result)


def function_name_generator_test() -> None:
    """Process every request with one model and display parameter schemas.

    Each parameter row shows the declared name and type. Generating
    argument values is the next stage. The elapsed wall-clock time
    includes model setup, input loading, and all requests.
    """
    start = datetime.datetime.now()
    model_object = Small_LLM_Model()
    function_names = function_name_encoding_test(model_object)
    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )
    prompt_file = data_io.model_validation(
        "data/input/function_calling_tests.json", data_models.PromptTest
    )

    prompt_text = []
    # Finish selection and schema display for one request in each turn.
    for prompt_element in prompt_file:
        print(f"Prompt: {prompt_element.prompt}")

        prompt_text.append(prompt_element.prompt)
        selection_prompt = generators.build_selection_prompt(
            prompt_element.prompt, validated_functions
        )
        result_function_name = generators.generate_function_name(
            model_object, selection_prompt, function_names, 10
        )

        print(f"Function name: {result_function_name}")
        # The selected name is a string; function is a model with .name.
        for function in validated_functions:
            if function.name == result_function_name:
                # Keys are parameter names; values are parameter models.
                for (
                    parameter_name,
                    parameter_definition,
                ) in function.parameters.items():
                    print(
                        f"Parameter name ({parameter_name}): "
                        f"Parameter type ({parameter_definition.type})"
                    )
        print()

    end = datetime.datetime.now()
    print(f"Execution time: {end - start}")


function_name_generator_test()
