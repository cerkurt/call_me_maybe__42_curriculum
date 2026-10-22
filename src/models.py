#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Literal

# `functions_definition.json` → `list[FunctionDefinition]`
# `function_calling_tests.json` → `list[PromptTest]`
# `ooutput.json` → `list[FunctionCallResult]`


# every value of key-value pair in parameters dict
class ParameterDefinition(BaseModel):
    type: Literal["number", "string", "boolean"]


# every object in returns dict
class ReturnDefinition(BaseModel):
    type: Literal["number", "string", "boolean"]


# every function object in function_definitions.json file
class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


# every prompt object in function_calling_tests.json file
class PromptTest(BaseModel):
    prompt: str


# every result object in output.json file (or output array)
class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, object]
