from llm_sdk import Small_LLM_Model

import json


def llm_model_test() -> None:
    model_object = Small_LLM_Model()

    test_str = "Hello LLM"
    test_result = model_object.encode(test_str)

    print("Encoded text result")
    print(type(test_result))
    print(test_result)
    print(test_result.shape)
    print()

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


llm_model_test()
constrained_decoding_test()
