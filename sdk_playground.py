"""Manual SDK experiments, selected by the function call at the bottom.

Run from the project root with: uv run python -m sdk_playground.
The active experiment prints selected names and their parameter schemas.
"""

from llm_sdk import Small_LLM_Model
from src import (
    data_io,
    data_models,
    llm_progress_builder,
    parameter_extractors as param_ext,
    llm_helpers,
    function_selectors,
    old_function_helpers,
    regex_helpers,
)

import json
import datetime
import re


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
    model_object = Small_LLM_Model()
    result1 = old_function_helpers.generate_text(
        model_object, "this is my start text", 5
    )

    result2 = old_function_helpers.generate_text(
        model_object, "this is my start text", 10
    )

    print(result1)
    print(result2)


def prompt_builder_test() -> None:
    """Display selection instructions built from validated definitions."""
    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )
    test_result = function_selectors.build_function_selector_prompt(
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
    promp_selection = function_selectors.build_function_selector_prompt(
        "Greet shrek", validated_functions
    )

    test_result = old_function_helpers.generate_text(
        model_object, promp_selection, 20
    )

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

    return name_id_match


def id_finder_test() -> None:
    """Inspect allowed next IDs for one manually specified name prefix."""
    model_object = Small_LLM_Model()
    test_dict = function_name_encoding_test(model_object)
    result_list = llm_helpers.find_allowed_next_token_ids(
        test_dict, [8822, 3062, 39794, 12993]
    )

    print(result_list)


def token_id_selector_test() -> None:
    """Trigger the selector's ValueError for an empty allowed-ID list."""
    result = llm_helpers.select_next_token_id([9.0, 7.0, 5.0], [])

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
        selection_prompt = function_selectors.build_function_selector_prompt(
            prompt_element.prompt, validated_functions
        )
        result_function_name = llm_helpers.generate_name_from_candidates(
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


def parameter_prompt_tester() -> None:
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
        prompt_text.append(prompt_element.prompt)
        selection_prompt = function_selectors.build_function_selector_prompt(
            prompt_element.prompt, validated_functions
        )
        result_function_name = llm_helpers.generate_name_from_candidates(
            model_object, selection_prompt, function_names, 10
        )

        for function in validated_functions:
            if function.name == result_function_name:
                # Keys are parameter names; values are parameter models.
                function_info = old_function_helpers.build_parameter_prompt(
                    prompt_element.prompt, function
                )
                result = old_function_helpers.generate_text(
                    model_object, function_info, 30
                )
                print(result)

    end = datetime.datetime.now()
    print(f"Execution time: {end - start}")


def string_state_tester() -> None:
    print(old_function_helpers.get_json_string_state('\t"abc"\n'))


def allowed_piece_tester() -> None:
    check = old_function_helpers.is_this_allowed_string_piece('"hel', "lo")
    print(check)


def first_sdk_tester() -> None:
    model_object = Small_LLM_Model()
    generated_ids = model_object.encode('"hel')[0].tolist()
    # print(f"Generated ids list start : {generated_ids}\n")

    candidate_token_ids: list = []
    file_path = model_object.get_path_to_vocab_file()
    with open(file_path, "r") as json_file:
        #  keys token texts, values are ids.
        file_content = json.load(json_file)
        for token_text, token_id in file_content.items():
            candidate_token_ids.append(token_id)

    allowed_ids = old_function_helpers.find_allowed_string_token_ids(
        model_object, generated_ids, candidate_token_ids
    )
    test_text_logits = model_object.get_logits_from_input_ids(generated_ids)
    next_token_id = llm_helpers.select_next_token_id(
        test_text_logits, allowed_ids
    )
    if next_token_id in allowed_ids:
        generated_ids.append(next_token_id)
        print(model_object.decode(generated_ids))
        print(
            old_function_helpers.get_json_string_state(
                model_object.decode(generated_ids)
            )
        )


def json_string_generator_tester() -> None:
    model_object = Small_LLM_Model()

    candidate_token_ids: list = []
    file_path = model_object.get_path_to_vocab_file()
    with open(file_path, "r") as json_file:
        #  keys token texts, values are ids.
        file_content = json.load(json_file)
        for token_text, token_id in file_content.items():
            candidate_token_ids.append(token_id)

    system_message = llm_helpers.format_request_message(
        "system",
        "You fill a function argument using the user request. "
        "Return the argument VALUE, never its name or type. "
        "Your entire response must be a JSON string literal "
        "(a quoted value),not a JSON object. Copy spelling and case "
        "from the request.",
    )

    user_message = llm_helpers.format_request_message(
        "user",
        "Function: fn_greet\n"
        "Description: Generate a greeting message for a person by name.\n"
        "Argument: name\n"
        "Type: string\n"
        "Request: Greet alice\n"
        "Return only the value.\n",
    )

    assistant_message = llm_helpers.format_request_message(
        "assistant", '"alice"'
    )

    user_request_prompt = "Greet Melda"
    request_text = llm_helpers.format_request_message(
        "user",
        f"Function: fn_greet\n"
        f"Description: Generate a greeting message for a person by name.\n"
        f"Argument: name\n"
        f"Type: string\n"
        f"Request: {user_request_prompt}\n"
        f"Return only the value.\n",
    )

    open_assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    model_input_text = (
        f"{system_message}\n"
        f"{user_message}\n"
        f"{assistant_message}\n"
        f"{request_text}\n"
        f"{open_assistant_start}\n"
    )
    result_output = old_function_helpers.generate_json_string(
        model_object, model_input_text, candidate_token_ids, 20
    )

    json_to_python_str = json.loads(result_output)
    if json_to_python_str not in user_request_prompt:
        raise ValueError("Generated value is not part of the user request.")
    result_dict: dict = {"name": json_to_python_str}

    function_calling_test = data_models.FunctionCallResult(
        prompt=user_request_prompt, name="fn_greet", parameters=result_dict
    )

    path = (
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/project_repo"
        "/sdk_test_write.json"
    )
    data_io.write_results(path, [function_calling_test])


def message_formatter_tester() -> None:
    print(llm_helpers.format_request_message("user", "Greet shrek"))


def string_slicer_tester() -> None:
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

    list_to_write: list = []
    for prompt_element in prompt_file:
        user_request = prompt_element.prompt
        selection_prompt = function_selectors.build_function_selector_prompt(
            user_request, validated_functions
        )
        result_function_name = llm_helpers.generate_name_from_candidates(
            model_object, selection_prompt, function_names, 10
        )

        for function in validated_functions:
            if function.name == result_function_name:
                parameter_dict: dict = {}
                for (
                    parameter_name,
                    parameter_type,
                ) in function.parameters.items():
                    candidates: list[float] | list[str]
                    if parameter_type.type == "number":
                        candidates = (
                            param_ext.extract_numbers_from_user_request(
                                user_request
                            )
                        )
                        model_input_text = (
                            param_ext.message_for_parameter_extraction(
                                user_request,
                                function,
                                parameter_name,
                                parameter_dict,
                            )
                        )

                    elif parameter_type.type == "string":
                        candidates = param_ext.string_slicer(user_request)
                        model_input_text = (
                            param_ext.message_for_parameter_extraction(
                                user_request,
                                function,
                                parameter_name,
                                parameter_dict,
                            )
                        )

                    json_and_id_match_dict: dict = {}
                    for candidate in candidates:
                        json_text = " " + json.dumps(candidate)
                        id_list = model_object.encode(json_text)[0].tolist()
                        json_and_id_match_dict.update({json_text: id_list})

                    result = llm_helpers.generate_name_from_candidates(
                        model_object,
                        model_input_text,
                        json_and_id_match_dict,
                        20,
                    )

                    parameter_value = json.loads(result)
                    parameter_dict.update({parameter_name: parameter_value})

                data_models.parameter_type_checker(function, parameter_dict)
                new_function_object = data_models.FunctionCallResult(
                    prompt=user_request,
                    name=function.name,
                    parameters=parameter_dict,
                )
                list_to_write.append(new_function_object)

    data_io.write_results(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/project_repo/"
        "write_test_io.json",
        list_to_write,
    )
    end = datetime.datetime.now()
    print(f"Execution time: {end - start}")


def number_extractor_tester() -> None:
    test_cases = [
        ("Set temperature to 18 degrees", [18.0]),
        ("Set temperature to -5.5 degrees", [-5.5]),
        ("Set temperature to .5 degrees", [0.5]),
        ("Set temperature to -.5 degrees", [-0.5]),
        ("Set temperature to 1e3 degrees", [1000.0]),
        ("Set temperature to 1,000 degrees", [1000.0]),
        ("Add 0.5 and -0.5", [0.5, -0.5]),
        ("Add -12.5 and 0.75", [-12.5, 0.75]),
        ("Add 1,234,567.89 and 2", [1234567.89, 2.0]),
        ("Add 2E-3 and 10", [0.002, 10.0]),
    ]

    for request, expected in test_cases:
        actual = param_ext.extract_numbers_from_user_request(request)

        print(f"Request: {request}")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")
        print(f"Result: {'PASS' if actual == expected else 'FAIL'}")
        print()


def number_messager_tester() -> None:
    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )
    for function in validated_functions:
        if function.name == "fn_add_numbers":
            print(
                param_ext.message_for_parameter_extraction(
                    "What is the sum of 2 and 3?", function, "b", {"a": 2.0}
                )
            )


def progress_builder_bool_tester(
    model_object: Small_LLM_Model,
    function_list: list[data_models.FunctionDefinition],
    parameter_name: str,
    expected_result_list: list[bool],
) -> None:
    prompt_list = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/project_repo/"
        "data/input/boolean_prompts.json",
        data_models.PromptTest,
    )

    result_list = llm_progress_builder.progress_builder(
        model_object, function_list, prompt_list
    )

    succesful_outputs: int = 0
    if len(expected_result_list) == len(result_list):
        for result, expected in zip(result_list, expected_result_list):
            output_result = result.parameters[parameter_name]
            print(
                f"Prompt: {result.prompt}\n"
                f"Expected result: {expected}\n"
                f"Actual result: {output_result}\n"
            )
            if output_result == expected:
                succesful_outputs += 1
    else:
        if len(expected_result_list) > len(result_list):
            raise ValueError("Missing function object(s) in the result list.")
        else:
            raise ValueError(
                "Missing boolean results to compare in expected results."
            )

    print(f"{succesful_outputs} / {len(expected_result_list)}")


def parameter_count_validate() -> None:
    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )
    for function in validated_functions:
        if function.name == "fn_add_numbers":
            data_models.parameter_type_checker(function, {"a": 21.55, "b": 3})
            print("success")


def writer_tester() -> None:
    path = "data/output/dummy.json"
    model_object = data_models.FunctionCallResult(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={"a": float("nan"), "b": 3.0},
    )
    data_io.write_results(path, [model_object])


def regex_tester() -> None:
    model_object = Small_LLM_Model()
    test_cases = [
        ("Replace all numbers in 'Room 12' with X", "Room 12", "X"),
        (
            "Replace all vowels in 'Programming is fun' with asterisks",
            "Programming is fun",
            "*",
        ),
        (
            "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat "
            "with another cat'",
            "The cat sat on the mat with another cat",
            "dog",
        ),
    ]

    for user_request, source_string, replacement in test_cases:
        selected_pattern = regex_helpers.generate_regex_pattern(
            model_object, user_request, source_string, replacement
        )

        print(re.sub(selected_pattern, replacement, source_string))


def regex_integration_tester() -> None:
    model_object = Small_LLM_Model()

    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )
    validated_prompts = data_io.model_validation(
        "data/input/function_calling_tests.json", data_models.PromptTest
    )

    results = llm_progress_builder.progress_builder(
        model_object, validated_functions, validated_prompts
    )

    for output in results:
        if output.name == "fn_substitute_string_with_regex":
            user_request = output.prompt
            source_string = output.parameters["source_string"]
            replacement = output.parameters["replacement"]

            if isinstance(source_string, str) and isinstance(replacement, str):
                generated_pattern = regex_helpers.generate_regex_pattern(
                    model_object, user_request, source_string, replacement
                )

                output.parameters["regex"] = generated_pattern

    data_io.write_results(
        "data/output/regex_integration_results.json", results
    )


def regex_converter_tester() -> None:
    model_object = Small_LLM_Model()

    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )
    validated_prompts = data_io.model_validation(
        "data/input/function_calling_tests.json", data_models.PromptTest
    )

    results = llm_progress_builder.progress_builder(
        model_object, validated_functions, validated_prompts
    )

    for output in results:
        if output.name == "fn_substitute_string_with_regex":
            user_request = output.prompt
            source_string = output.parameters["source_string"]
            replacement = output.parameters["replacement"]

            if isinstance(source_string, str) and isinstance(replacement, str):
                replacement = regex_helpers.convert_regex_replacement_value(
                    user_request, replacement
                )
                output.parameters["replacement"] = replacement

                generated_pattern = regex_helpers.generate_regex_pattern(
                    model_object, user_request, source_string, replacement
                )
                output.parameters["regex"] = generated_pattern

                print(
                    f"User request: {user_request}\n"
                    f"Source string: {source_string}\n"
                    f"Replacement: {replacement}\n"
                    f"Generated pattern: {generated_pattern}"
                )

                print(
                    re.sub(generated_pattern, replacement, source_string),
                    end="\n\n",
                )

    print(
        regex_helpers.convert_regex_replacement_value(
            "Replace all vowels with asterisks", "asterisks"
        )
    )

    print(
        regex_helpers.convert_regex_replacement_value(
            "Replace all vowels with 'asterisks'", "asterisks"
        )
    )


def candidate_pool_tester() -> None:
    requests = [
        "Greet Ceren Kurt",
        "Echo the text Good morning everyone",
        "Greet Ceren using the word 'hello'",
        "Greet Ceren Kurt using the word 'hello'",
    ]

    model_object = Small_LLM_Model()
    validated_functions = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_functions.json",
        data_models.FunctionDefinition,
    )

    prompts = []
    for request in requests:
        prompts.append(data_models.PromptTest(prompt=request))

    result_list = llm_progress_builder.progress_builder(
        model_object,
        validated_functions,
        prompts,
    )

    for request, result in zip(requests, result_list):
        candidates = param_ext.string_slicer(request)

        print(f"Request: {request}")
        print(f"Candidates: {candidates}")
        print(f"Selected function: {result.name}")
        print(f"Selected parameters: {result.parameters}\n")


def string_selection_tester() -> None:
    start = datetime.datetime.now()

    model_object = Small_LLM_Model()
    validated_functions = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_functions.json",
        data_models.FunctionDefinition,
    )
    validated_prompts = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_prompts.json",
        data_models.PromptTest,
    )

    result_list = llm_progress_builder.progress_builder(
        model_object, validated_functions, validated_prompts
    )

    data_io.write_results(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/project_repo/"
        "data/output/extended_test_results.json",
        result_list,
    )

    end = datetime.datetime.now()
    print(f"Execution time: {end - start}")


def string_parameter_selection_tester() -> None:
    model_object = Small_LLM_Model()

    functions = {
        "echo_text": data_models.FunctionDefinition(
            name="echo_text",
            description="Return the provided text unchanged.",
            parameters={
                "text": data_models.ParameterDefinition(type="string")
            },
            returns=data_models.ReturnDefinition(type="string"),
        ),
        "fn_greet": data_models.FunctionDefinition(
            name="fn_greet",
            description="Generate a greeting for a person by name.",
            parameters={
                "name": data_models.ParameterDefinition(type="string")
            },
            returns=data_models.ReturnDefinition(type="string"),
        ),
    }

    test_cases = [
        (
            "Greet Ceren using the word 'hello'",
            "fn_greet",
            "name",
            "Ceren",
        ),
    ]

    for request, function_name, parameter_name, expected in test_cases:
        function = functions[function_name]

        candidates = param_ext.string_slicer(request)

        model_input_text = param_ext.message_for_parameter_extraction(
            request,
            function,
            parameter_name,
            {},
        )

        candidate_id_dict = {}
        for candidate in candidates:
            json_text = " " + json.dumps(candidate, ensure_ascii=False)
            candidate_id_dict[json_text] = model_object.encode(json_text)[
                0
            ].tolist()

        selected = llm_helpers.generate_name_from_candidates(
            model_object,
            model_input_text,
            candidate_id_dict,
            50,
        )

        actual = json.loads(selected)

        print("=" * 60)
        print(f"REQUEST: {request}")
        print(f"PARAMETER: {parameter_name}")
        print(f"CANDIDATES: {candidates}")
        print(f"EXPECTED IN POOL: {expected in candidates}")

        print("\nMODEL INPUT:")
        print(model_input_text)

        print(f"\nEXPECTED: {expected!r}")
        print(f"ACTUAL:   {actual!r}")
        print(f"RESULT: {'PASS' if actual == expected else 'FAIL'}")
        print()


def notification_function_selection_tester() -> None:
    model_object = Small_LLM_Model()

    validated_functions = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_functions.json",
        data_models.FunctionDefinition,
    )

    name_id_dict = old_function_helpers.function_name_encoder(
        model_object, validated_functions
    )

    test_cases = [
        (
            "Do not enable notifications for room 'office'.",
            "set_notifications",
        ),
        (
            "Don't turn on notifications for room 'office'.",
            "set_notifications",
        ),
        (
            "Keep notifications disabled in the office.",
            "set_notifications",
        ),
        (
            "I don't want notifications enabled in the kitchen.",
            "set_notifications",
        ),
        (
            "Prevent notifications from being enabled in the bedroom.",
            "set_notifications",
        ),
        (
            "Please ensure the office notifications remain off.",
            "set_notifications",
        ),
    ]

    for request, expected in test_cases:
        selection_prompt = function_selectors.build_function_selector_prompt(
            request, validated_functions
        )

        selected = llm_helpers.generate_name_from_candidates(
            model_object,
            selection_prompt,
            name_id_dict,
            10,
        )

        expected = "set_notifications"

        print("=" * 60)
        print(f"REQUEST:  {request}")
        print(f"EXPECTED: {expected}")
        print(f"ACTUAL:   {selected}")
        print(f"RESULT:   {'PASS' if selected == expected else 'FAIL'}")

        if selected != expected:
            print("\nMODEL INPUT:")
            print(selection_prompt)


def function_selection_trace_tester() -> None:
    model_object = Small_LLM_Model()

    validated_functions = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_functions.json",
        data_models.FunctionDefinition,
    )

    name_id_dict = old_function_helpers.function_name_encoder(
        model_object, validated_functions
    )

    requests = [
        "Do not enable notifications for room 'office'.",
        "Make sure notifications stay disabled for room 'office'.",
    ]

    for request in requests:
        print("\n" + "=" * 70)
        print("REQUEST:", request)

        selection_prompt = function_selectors.build_function_selector_prompt(
            request, validated_functions
        )

        all_token_ids = model_object.encode(selection_prompt)[0].tolist()
        generated_token_ids: list = []

        for step in range(10):
            completed = [
                name
                for name, ids in name_id_dict.items()
                if ids == generated_token_ids
            ]

            if completed:
                print("FINAL SELECTION:", completed[0])
                break

            allowed_ids = llm_helpers.find_allowed_next_token_ids(
                name_id_dict, generated_token_ids
            )

            logits = model_object.get_logits_from_input_ids(all_token_ids)

            ranked_ids = sorted(
                allowed_ids,
                key=lambda token_id: logits[token_id],
                reverse=True,
            )

            print(f"\nSTEP {step + 1}")

            for token_id in ranked_ids:
                possible_functions = [
                    name
                    for name, ids in name_id_dict.items()
                    if ids[: len(generated_token_ids) + 1]
                    == generated_token_ids + [token_id]
                ]

                print(
                    f"Token: {model_object.decode([token_id])!r} "
                    f"| Logit: {logits[token_id]:.4f} "
                    f"| Functions: {possible_functions}"
                )

            selected_id = ranked_ids[0]
            generated_token_ids.append(selected_id)
            all_token_ids.append(selected_id)

        else:
            print("MAX TOKEN COUNT REACHED")


def function_label_selection_tester() -> None:
    model_object = Small_LLM_Model()

    validated_functions = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_functions.json",
        data_models.FunctionDefinition,
    )

    orders = {
        "original": validated_functions,
        "reversed": list(reversed(validated_functions)),
        "rotated": validated_functions[1:] + validated_functions[:1],
    }

    test_cases = [
        (
            "Do not enable notifications for room 'office'.",
            "set_notifications",
        ),
        ("Greet Ceren Kurt", "fn_greet"),
    ]

    for order_name, ordered_functions in orders.items():

        label_to_function = {
            chr(65 + index): function
            for index, function in enumerate(ordered_functions)
        }

        label_to_ids = {
            label: model_object.encode(label)[0].tolist()
            for label in label_to_function
        }

        options = "\n".join(
            f"{label}: {function.name} - {function.description}"
            for label, function in label_to_function.items()
        )

        print(f"\nORDER: {order_name}")
        print("LABEL MAPPING:")
        for label, function in label_to_function.items():
            print(f"{label}: {function.name}")

        for request, expected in test_cases:
            system_message = llm_helpers.format_request_message(
                "system",
                "Choose the available function that best matches "
                "the user's request. Consider the intended action "
                "and its meaning, including negation. "
                "Return only the single-letter label of the "
                "correct function. Do not execute it.",
            )

            user_message = llm_helpers.format_request_message(
                "user",
                f"Available functions:\n{options}\n\n"
                f"Request: {request}\n"
                "Which label identifies the correct function?",
            )

            assistant_prefix = (
                "<|im_start|>assistant\n" "<think>\n\n</think>\n\n"
            )

            selection_prompt = system_message + user_message + assistant_prefix

            selected_label = llm_helpers.generate_name_from_candidates(
                model_object,
                selection_prompt,
                label_to_ids,
                10,
            )

            selected_function = label_to_function[selected_label].name

            print("=" * 60)
            print(f"REQUEST:  {request}")
            print(f"LABEL:    {selected_label}")
            print(f"EXPECTED: {expected}")
            print(f"ACTUAL:   {selected_function}")
            print(
                f"RESULT:   "
                f"{'PASS' if selected_function == expected else 'FAIL'}"
            )


def full_function_scoring_tester() -> None:
    model_object = Small_LLM_Model()

    validated_functions = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_functions.json",
        data_models.FunctionDefinition,
    )

    requests = [
        "Do not disable notifications for room 'office'.",
        "Do not enable notifications for room 'office'.",
        "Don't turn off notifications for room 'office'.",
        "Keep notifications enabled for room 'office'.",
        "Greet Ceren Kurt",
        "Echo 'Çağrı 🌡️'",
    ]

    for request in requests:
        print(f"\nREQUEST: {request}")
        print(param_ext.string_slicer(request))

        selection_prompt = function_selectors.build_function_selector_prompt(
            request, validated_functions
        )

        scores = {}

        for function in validated_functions:
            score = llm_helpers.score_function_candidate(
                model_object,
                selection_prompt,
                function.name,
            )
            scores[function.name] = score

        for name, score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"{name}: {score:.4f}")


def string_slicer_tester_new() -> None:
    requests = [
        "Greet Shrek",
        "Greet shrek",
        "Greet Melinda Selinda using the word 'hello'",
    ]

    for request in requests:
        print(f"\nREQUEST: {request}")
        print("CANDIDATES:", param_ext.string_slicer(request))


def string_candidate_scoring_tester() -> None:
    model_object = Small_LLM_Model()

    validated_functions = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_functions.json",
        data_models.FunctionDefinition,
    )

    function = next(
        function
        for function in validated_functions
        if function.name == "fn_greet"
    )

    test_cases = [
        (
            "Greet Melinda Selinda using the word 'hello'",
            "Melinda Selinda",
        ),
        (
            "Greet Ceren Kurt using the word 'hello'",
            "Ceren Kurt",
        ),
        (
            "Greet Shrek",
            "Shrek",
        ),
        (
            "Greet Captain Jack Sparrow using the word 'hello'",
            "Captain Jack Sparrow",
        ),
    ]

    for request, expected in test_cases:
        candidates = param_ext.string_slicer(request)

        model_input_text = param_ext.message_for_parameter_extraction(
            request,
            function,
            "name",
            {},
        )

        scores = []

        for candidate in candidates:
            json_text = " " + json.dumps(
                candidate,
                ensure_ascii=False,
            )

            score = llm_helpers.score_function_candidate(
                model_object,
                model_input_text,
                json_text,
            )

            token_count = len(model_object.encode(json_text)[0].tolist())
            average_score = score / token_count

            scores.append((candidate, score, average_score, token_count))

        scores.sort(key=lambda item: item[2], reverse=True)

        print("=" * 60)
        print(f"REQUEST: {request}")
        print(f"EXPECTED: {expected!r}")

        print("\nTOP 5 BY AVERAGE SCORE:")
        for candidate, total, average, count in scores[:5]:
            print(
                f"{candidate!r}: "
                f"total={total:.4f}, "
                f"average={average:.4f}, "
                f"tokens={count}"
            )


def room_selection_debug_tester() -> None:
    requests = [
        "Enable notifications for room 'office'.",
        "Don't turn on notifications for room 'office'.",
        "Don't turn off notifications for room 'office'.",
        "I'm asking you to greet 'Ceren'.",
        'Echo "I\'m Ceren"',
    ]

    for request in requests:
        print(f"\nREQUEST: {request}")
        print(
            "QUOTED:",
            param_ext.string_splitter_inside_the_quotes(request),
        )
        print("CANDIDATES:", param_ext.string_slicer(request))


def function_capability_tester() -> None:
    model_object = Small_LLM_Model()

    functions = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_functions.json",
        data_models.FunctionDefinition,
    )

    validated_prompts = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_prompts.json",
        data_models.PromptTest,
    )

    # Expected function names for this specific 42-prompt test fixture.
    expected_names = (
        ["set_temperature"] * 6
        + ["set_notifications"] * 5
        + ["echo_text"] * 6
        + ["fn_add_numbers"] * 6
        + ["fn_reverse_string"] * 3
        + ["fn_greet"] * 16
    )

    test_cases = [
        (prompt.prompt, expected)
        for prompt, expected in zip(
            validated_prompts, expected_names, strict=True
        )
    ]

    for request, expected in test_cases:
        scores = {}

        for function in functions:
            parameters = ", ".join(
                f"{name}: {definition.type}"
                for name, definition in function.parameters.items()
            )

            prompt = (
                llm_helpers.format_request_message(
                    "system",
                    "Determine whether the given function can fulfill "
                    "the user's request by choosing appropriate argument "
                    "values. Consider the function's actual capability, "
                    "not whether the user wants an action enabled or "
                    "disabled. Answer only Yes or No.",
                )
                + llm_helpers.format_request_message(
                    "user",
                    f"Request: {request}\n"
                    f"Function: {function.name}\n"
                    f"Description: {function.description}\n"
                    f"Parameters: {parameters}\n\n"
                    "Can this function fulfill the request?",
                )
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )

            yes_score = llm_helpers.score_function_candidate(
                model_object, prompt, "Yes"
            )

            no_score = llm_helpers.score_function_candidate(
                model_object, prompt, "No"
            )

            scores[function.name] = yes_score - no_score

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        actual = ranked[0][0]
        margin = ranked[0][1] - ranked[1][1]

        print(
            f"{'PASS' if actual == expected else 'FAIL'} | "
            f"Expected: {expected} | "
            f"Actual: {actual} | "
            f"Margin: {margin:.4f} | "
            f"Request: {request}"
        )


def pairwise_function_tester() -> None:
    model_object = Small_LLM_Model()

    functions = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_functions.json",
        data_models.FunctionDefinition,
    )

    validated_prompts = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_prompts.json",
        data_models.PromptTest,
    )

    expected_names = (
        ["set_temperature"] * 6
        + ["set_notifications"] * 5
        + ["echo_text"] * 6
        + ["fn_add_numbers"] * 6
        + ["fn_reverse_string"] * 3
        + ["fn_greet"] * 16
    )

    test_cases = [
        (prompt.prompt, expected)
        for prompt, expected in zip(
            validated_prompts, expected_names, strict=True
        )
    ]

    correct = 0

    for request, expected in test_cases:
        actual = function_selectors.hybrid_function_selector(
            model_object, request, functions
        )

        passed = actual == expected
        correct += passed

        print(
            f"{'PASS' if passed else 'FAIL'} | "
            f"Expected: {expected} | "
            f"Actual: {actual} | "
            f"Request: {request}"
        )

    print(f"\nTOTAL: {correct}/{len(test_cases)}")


def debug_string_problems():
    model_object = Small_LLM_Model()

    functions = data_io.model_validation(
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/test_inputs/"
        "mandatory_functions.json",
        data_models.FunctionDefinition,
    )

    greet_function = None

    for function in functions:
        if function.name == "fn_greet":
            greet_function = function
            break

    requests = [
        "Greet the person by name and surname: Ceren Kurt",
        "Say hello to my best friend Kenny.",
    ]

    values_to_check = [
        ["Ceren", "Ceren Kurt"],
        ["Kenny", "Kenny."],
    ]

    for index, request in enumerate(requests):
        print("\nREQUEST:", request)

        candidates = llm_progress_builder.string_slicer(request)

        parameter_prompt = (
            llm_progress_builder.message_for_parameter_extraction(
                request,
                greet_function,
                "name",
                {},
            )
        )

        for candidate in values_to_check[index]:
            if candidate not in candidates:
                print(candidate, "-> NOT IN CANDIDATES")
                continue

            json_candidate = " " + json.dumps(
                candidate,
                ensure_ascii=False,
            )

            score = llm_helpers.score_function_candidate(
                model_object,
                parameter_prompt,
                json_candidate,
            )

            token_ids = model_object.encode(json_candidate)[0].tolist()

            average_score = score / len(token_ids)

            print(
                candidate,
                "tokens:",
                len(token_ids),
                "total:",
                round(score, 4),
                "average:",
                round(average_score, 4),
            )
            print(candidate, "-> SCORE:", round(score, 4))
