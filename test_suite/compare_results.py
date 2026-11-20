"""Compare expected calls with saved CLI output without loading an LLM."""

import argparse
import json


class C:
    """ANSI color codes used to make terminal output easier to read."""

    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    PURPLE = "\033[35m"
    BG_RED = "\033[41m"


def compare_results() -> None:
    parser = argparse.ArgumentParser(
        description="Compare expected and actual function calls."
    )
    parser.add_argument("--expected", required=True)
    parser.add_argument("--actual", required=True)
    args = parser.parse_args()

    with open(args.expected, "r", encoding="utf-8") as file:
        expected = json.load(file)

    with open(args.actual, "r", encoding="utf-8") as file:
        actual = json.load(file)

    correct_count = 0

    for index, expected_result in enumerate(expected):
        print(f"{C.YELLOW}\n===== Result {index + 1} ====={C.RESET}")

        if index >= len(actual):
            print("MISSING: No actual result.")
            print("Expected:", expected_result)
            continue

        try:
            assert actual[index] == expected_result
        except AssertionError:
            print("FAIL")
            print("Expected:", expected_result)
            print("Actual:  ", actual[index])

        else:
            print("PASS")
            correct_count += 1

    for index in range(len(expected), len(actual)):
        print(f"\nEXTRA result {index + 1}:")
        print(actual[index])

    missing_count = max(0, len(expected) - len(actual))
    extra_count = max(0, len(actual) - len(expected))

    print(f"\n{C.BLUE}===== Summary ====={C.RESET}")
    print(f"{C.GREEN}Correct:{C.RESET} {correct_count}/{len(expected)}")
    print(f"{C.RED}Missing results:{C.RESET} {missing_count}")
    print(f"Extra results: {extra_count}")

    if expected:
        accuracy = correct_count / len(expected) * 100
        print(f"Accuracy: {accuracy:.1f}%\n")


if __name__ == "__main__":
    compare_results()
