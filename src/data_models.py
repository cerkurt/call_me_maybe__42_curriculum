#!/usr/bin/env python3

"""Pydantic schemas for function definitions, prompt inputs, and results."""

from pydantic import BaseModel
from typing import Literal

# `functions_definition.json` → `list[FunctionDefinition]`
# `function_calling_tests.json` → `list[PromptTest]`
# Output JSON → `list[FunctionCallResult]`


class ParameterDefinition(BaseModel):
    """Declare the allowed JSON type for one function parameter."""

    type: Literal["number", "string", "boolean"]


class ReturnDefinition(BaseModel):
    """Declare the JSON type of a function's return value."""

    type: Literal["number", "string", "boolean"]


class FunctionDefinition(BaseModel):
    """Describe one available function and its parameter declarations.

    parameters is a dictionary: keys are parameter names and values are
    ParameterDefinition objects. Each value exposes its declared type
    through .type. The function's own name is stored in .name.
    """

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptTest(BaseModel):
    """Hold one user request from the input prompt array."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Hold a structured call with its original prompt and argument values.

    Parameter names and value types need separate checks against the
    selected FunctionDefinition; this model accepts arbitrary values.
    """

    prompt: str
    name: str
    parameters: dict[str, object]
