# JSON schema enforcer

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Example JSON Schema](#example-json-schema)
- [Potential Use Cases](#potential-use-cases)

## Introduction

json-schema-enforcer is a Python package that allows you to progressively validate a JSON string against a JSON schema. With this package, you can check if a JSON string adheres to the schema and if it is complete, even if the string is not fully formed. This is particularly useful when working with large language models to constraint text generation.

This project was created out of spite for the lack of a similar feature in [OpenAI ChatGPT's function calling](https://openai.com/blog/function-calling-and-other-api-updates), to prove that creating something like this is very easy. It uses the simplified JSON schema specification, similar to what OpenAI does. Please note that the code implementation probably does not adhere to best practices, as it is primarily an experimental showcase by someone who isn't skilled in creating programming language parsers.

## Key Features

- Parses a JSON schema and validates a string against it
- Determines if the returned JSON is valid (matches the schema and optional formatting)
- Determines if the JSON string is complete and provides the index of the last character

## Installation

You can install json-schema-enforcer using pip:

```shell
pip install json-schema-enforcer
```

## Usage

To use json-schema-enforcer in your Python project, follow these steps:

```python
import json_schema_enforcer

# Parse the schema
parser = json_schema_enforcer.parser_for_schema(schema)

# Check if the parser is valid
if parser is None:
    raise ValueError("Invalid JSON schema")

# Validate the JSON string
validated = parser.validate(maybe_json_string)

# Print the validation result
print(validated.valid)  # Whether it adheres to the schema
print("complete" if validated.end_index else "incomplete")
```

## Example JSON Schema

You can use the following example JSON schema as a reference when working with json-schema-enforcer:

```json
{
  "type": "object",
  "properties": {
    "name": { "type": "string", "maxLength": 20 },
    "age": { "type": "integer" },
    "preferred_animal": { "type": "string", "enum": ["cat", "dog"] }
  },
  "required": ["name", "age"]
}
```

## Potential Use Cases

- Constraining text generation with large language models; example: [local-llm-function-calling](https://github.com/rizerphe/local-llm-function-calling), a project that allows you to constrain generation with any huggingface model

