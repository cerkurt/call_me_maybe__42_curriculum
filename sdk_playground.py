"""Manual SDK experiments, selected by the function call at the bottom.

Run from the project root with: uv run python -m sdk_playground.
The active experiment prints selected names and their parameter schemas.
"""

from llm_sdk import Small_LLM_Model
from src import data_io, data_models, generators

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
    result1 = generators.generate_text(
        model_object, "this is my start text", 5
    )

    result2 = generators.generate_text(
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
        result_function_name = generators.generate_name_from_candidates(
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
        selection_prompt = generators.build_selection_prompt(
            prompt_element.prompt, validated_functions
        )
        result_function_name = generators.generate_name_from_candidates(
            model_object, selection_prompt, function_names, 10
        )

        for function in validated_functions:
            if function.name == result_function_name:
                # Keys are parameter names; values are parameter models.
                function_info = generators.build_parameter_prompt(
                    prompt_element.prompt, function
                )
                result = generators.generate_text(
                    model_object, function_info, 30
                )
                print(result)

    end = datetime.datetime.now()
    print(f"Execution time: {end - start}")


def string_state_tester() -> None:
    print(generators.get_json_string_state('\t"abc"\n'))


def allowed_piece_tester() -> None:
    check = generators.is_this_allowed_string_piece('"hel', "lo")
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

    # candidate_token_ids.extend(model_object.encode('"')[0].tolist())
    # candidate_token_ids.extend(model_object.encode("lo")[0].tolist())
    # candidate_token_ids.extend(model_object.encode("\n")[0].tolist())

    allowed_ids = generators.find_allowed_string_token_ids(
        model_object, generated_ids, candidate_token_ids
    )
    test_text_logits = model_object.get_logits_from_input_ids(generated_ids)
    next_token_id = generators.select_next_token_id(
        test_text_logits, allowed_ids
    )
    if next_token_id in allowed_ids:
        generated_ids.append(next_token_id)
        print(model_object.decode(generated_ids))
        print(
            generators.get_json_string_state(
                model_object.decode(generated_ids)
            )
        )
    # for token_id in candidate_token_ids:
    #     print(f"Token id: {token_id}")
    #     print(f"Candidate text: {repr(model_object.decode([token_id]))}")
    #     if token_id in allowed_ids:
    #         print("Candidate in alowed tokens: Yes\n")
    #     else:
    #         print("Candidate in allowed tokens: No\n")
    # print(f"Generated ids list end: {generated_ids}")


def json_string_generator_tester() -> None:
    model_object = Small_LLM_Model()

    candidate_token_ids: list = []
    file_path = model_object.get_path_to_vocab_file()
    with open(file_path, "r") as json_file:
        #  keys token texts, values are ids.
        file_content = json.load(json_file)
        for token_text, token_id in file_content.items():
            candidate_token_ids.append(token_id)

    # candidate_token_ids.extend(model_object.encode('"')[0].tolist())
    # candidate_token_ids.extend(model_object.encode("lo")[0].tolist())
    # candidate_token_ids.extend(model_object.encode("\n")[0].tolist())

    system_message = generators.format_request_message(
        "system",
        "You fill a function argument using the user request. "
        "Return the argument VALUE, never its name or type. "
        "Your entire response must be a JSON string literal "
        "(a quoted value),not a JSON object. Copy spelling and case "
        "from the request.",
    )

    user_message = generators.format_request_message(
        "user",
        "Function: fn_greet\n"
        "Description: Generate a greeting message for a person by name.\n"
        "Argument: name\n"
        "Type: string\n"
        "Request: Greet alice\n"
        "Return only the value.\n",
    )

    assistant_message = generators.format_request_message(
        "assistant", '"alice"'
    )

    user_request_prompt = "Greet Melda"
    request_text = generators.format_request_message(
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
    result_output = generators.generate_json_string(
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

    # print(result_dict)
    # print(repr(json_to_python_str))
    # print(repr(result_output))
    # print(generators.get_json_string_state(result_output))


def message_formatter_tester() -> None:
    print(generators.format_request_message("user", "Greet shrek"))


def string_parameter_messages_tester() -> None:
    model_object = Small_LLM_Model()

    candidate_token_ids: list = []
    file_path = model_object.get_path_to_vocab_file()
    with open(file_path, "r") as json_file:
        #  keys token texts, values are ids.
        file_content = json.load(json_file)
        for token_text, token_id in file_content.items():
            candidate_token_ids.append(token_id)

    path = (
        "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/"
        "project_repo/data/input/functions_definition.json"
    )
    function_definitions = data_io.model_validation(
        path, data_models.FunctionDefinition
    )

    for function in function_definitions:
        if function.name == "fn_reverse_string":
            model_input_text = generators.message_for_string_parameter(
                "Reverse the string 'hello'", function, "s"
            )
    json_text_output = generators.generate_json_string(
        model_object, model_input_text, candidate_token_ids, 10
    )

    print(json.loads(json_text_output))


def string_slicer_tester() -> None:
    # user_request = "Reverse 'hello world' please"
    # parameter_name = "s"
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
        selection_prompt = generators.build_selection_prompt(
            user_request, validated_functions
        )
        result_function_name = generators.generate_name_from_candidates(
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
                            generators.extract_numbers_from_user_request(
                                user_request
                            )
                        )
                        model_input_text = (
                            generators.message_for_number_parameter(
                                user_request,
                                function,
                                parameter_name,
                                parameter_dict,
                            )
                        )

                    elif parameter_type.type == "string":
                        candidates = generators.string_slicer(
                            user_request, len(function.parameters)
                        )
                        model_input_text = (
                            generators.message_for_number_parameter(
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

                    result = generators.generate_name_from_candidates(
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
    print(generators.extract_numbers_from_user_request("21.5 and -5.5 and 0"))


def number_messager_tester() -> None:
    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )
    for function in validated_functions:
        if function.name == "fn_add_numbers":
            print(
                generators.message_for_number_parameter(
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

    result_list = generators.progress_builder(
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


# model_object = Small_LLM_Model()

# progress_builder_bool_tester(
#     model_object,
#     data_io.model_validation(
#         "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/"
#         "project_repo/data/input/boolean_enabled_functions.json",
#         data_models.FunctionDefinition,
#     ),
#     "enabled",
#     [True, False],
# )

# progress_builder_bool_tester(
#     model_object,
#     data_io.model_validation(
#         "/Users/ceren/Desktop/Codam/Milestone_03/call_me_maybe/"
#         "project_repo/data/input/boolean_disabled_functions copy.json",
#         data_models.FunctionDefinition,
#     ),
#     "disabled",
#     [False, True],
# )


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
        selected_pattern = generators.generate_regex_pattern(
            model_object, user_request, source_string, replacement
        )

        print(re.sub(selected_pattern, replacement, source_string))


# # selected_pattern = generators.generate_name_from_candidates(
# #     model_object, selection_message, pattern_dict, 10
# # )

# selected_label = generators.generate_name_from_candidates(
#     model_object, selection_message, label_dict, 10
# )
# selected_pattern = available_patterns[selected_label]

# print(selected_label, selected_pattern)
# print(re.sub(selected_pattern, replacement, source_string))

# system_message = (
#     "Return only the requested input argument value as one "
#     "complete JSON string. Do not execute the function. Only replace the"
#     "requested part that is described in the user request with the "
#     "given replacement. Do not try to match with the other parts of the "
#     "request."
# )

# example_user_message = (
#     "Function: find_matches\n"
#     "Description: Find text matching a regular expression.\n"
#     "Argument: pattern\n"
#     "Type: string\n"
#     "Request: Find uppercase letter sequences in Hello NASA.\n"
#     "Return only the value of pattern.\n"
# )

# example_assistant_answer = '"[A-Z]+"'

# formatted_system_message = generators.format_request_message(
#     "system", system_message
# )
# formatted_example_user_message = generators.format_request_message(
#     "user", example_user_message
# )
# formatted_example_assistant_message = generators.format_request_message(
#     "assistant", example_assistant_answer
# )

# for function in validated_functions:
#     if function.name == "fn_substitute_string_with_regex":
#         user_message = (
#             f"Function: {function.name}\n"
#             f"Description: {function.description}\n"
#             f"Argument: {parameter_to_produce}\n"
#             f"Type: {function.parameters[parameter_to_produce].type}\n"
#             f"Request: {user_request}\n"
#             f"Return only the value of the {parameter_to_produce}.\n"
#         )

#         formatted_user_message = generators.format_request_message(
#             "user", user_message
#         )
#         model_input_text = (
#             formatted_system_message
#             + formatted_example_user_message
#             + formatted_example_assistant_message
#             + formatted_user_message
#             + open_assistant_start
#         )
#         print(model_input_text)

# json_string = generators.generate_json_string(
#     model_object, model_input_text, candidate_token_ids, 10
# )
# print(json_string)

# print(re.sub(json.loads(json_string), "*", "Programming is fun"))
# print(re.sub("[aeiouAEIOU]", "*", "Programming is fun"))
# print(re.sub("[a-zA-Z]+", "*", "Programming is fun"))


def regex_integration_tester() -> None:
    model_object = Small_LLM_Model()

    validated_functions = data_io.model_validation(
        "data/input/functions_definition.json",
        data_models.FunctionDefinition,
    )
    validated_prompts = data_io.model_validation(
        "data/input/function_calling_tests.json", data_models.PromptTest
    )

    results = generators.progress_builder(
        model_object, validated_functions, validated_prompts
    )

    for output in results:
        if output.name == "fn_substitute_string_with_regex":
            user_request = output.prompt
            source_string = output.parameters["source_string"]
            replacement = output.parameters["replacement"]

            if isinstance(source_string, str) and isinstance(replacement, str):
                generated_pattern = generators.generate_regex_pattern(
                    model_object, user_request, source_string, replacement
                )

                output.parameters["regex"] = generated_pattern

                # print(
                #     f"User request: {user_request}\n"
                #     f"Source string: {source_string} / "
                #     f"Replacement: {replacement} / "
                #     f"Generated pattern: {generated_pattern}\n"
                # )

                # print(re.sub(generated_pattern, replacement, source_string))

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

    results = generators.progress_builder(
        model_object, validated_functions, validated_prompts
    )

    for output in results:
        if output.name == "fn_substitute_string_with_regex":
            user_request = output.prompt
            source_string = output.parameters["source_string"]
            replacement = output.parameters["replacement"]

            if isinstance(source_string, str) and isinstance(replacement, str):
                replacement = generators.convert_regex_replacement_value(
                    user_request, replacement
                )
                output.parameters["replacement"] = replacement

                generated_pattern = generators.generate_regex_pattern(
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
        generators.convert_regex_replacement_value(
            "Replace all vowels with asterisks", "asterisks"
        )
    )

    print(
        generators.convert_regex_replacement_value(
            "Replace all vowels with 'asterisks'", "asterisks"
        )
    )
