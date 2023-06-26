"""A container for a JSON schema"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
import re
from typing import Iterator, Literal, Type


@dataclass
class StyleConfig:
    """The configuration for the style of the schema"""

    force_instant_comma: bool = True
    indent: int | None = 4
    compact_objects: bool = True
    max_inline_array_elements: int | None = 5
    object_multiline_treshold: int | None = 2


@dataclass
class ValidationResult:
    """The result of a validation"""

    valid: bool
    end_index: int | None = None


class JsonSchemaParser(ABC):
    """A generic JSON schema parser"""

    @classmethod
    @abstractmethod
    def load(
        cls, schema: dict, recursive_parsers: list[Type[JsonSchemaParser]]
    ) -> JsonSchemaParser | None:
        """Load the schema from a dict, returning None if it's invalid

        Args:
            schema: The schema to load
            recursive_parsers: The parsers to use for schemas contained within this one
        """

    @abstractmethod
    def validate(
        self,
        data: str,
        start_index: int = 0,
        indent_level: int = 0,
        style_config: StyleConfig | None = None,
    ) -> ValidationResult:
        """Validate the data against the schema

        Args:
            data: The data to validate
            start_index: The index the object starts at
            indent_level: The level of indentation of the current object
            style_config: The configuration for the style of the schema
        """

    @classmethod
    def parser_for(
        cls, schema: dict, parsers: list[Type[JsonSchemaParser]]
    ) -> JsonSchemaParser:
        """Get the parser for a schema"""
        for parser in parsers:
            loaded = parser.load(schema, parsers)
            if loaded is not None:
                return loaded
        raise ValueError(f"No parser for {schema}")


class LiteralValueSchemaParser(JsonSchemaParser):
    """A schema parser for the boolean and null types"""

    @classmethod
    def load(
        cls, schema: dict, _recursive_parsers: list[Type[JsonSchemaParser]]
    ) -> LiteralValueSchemaParser | None:
        if schema["type"] in ("boolean", "null"):
            return cls()
        return None

    def validate(
        self,
        data: str,
        start_index: int = 0,
        _indent_level: int = 0,
        _style_config: StyleConfig | None = None,
    ) -> ValidationResult:
        for literal_value in ("true", "false", "null"):
            parsed = data[start_index:]
            if parsed.startswith(literal_value, start_index):
                return ValidationResult(True, start_index + len(literal_value))
            if literal_value.startswith(parsed):
                # Means we are not finished parsing that one just yet
                # so no end index is returned
                return ValidationResult(True, None)
        return ValidationResult(False, None)


class IntegerSchemaParser(JsonSchemaParser):
    """A schema parser for integer values"""

    @classmethod
    def load(
        cls, schema: dict, _recursive_parsers: list[Type[JsonSchemaParser]]
    ) -> IntegerSchemaParser | None:
        if schema["type"] == "integer":
            return cls()
        return None

    def validate(
        self,
        data: str,
        start_index: int = 0,
        _indent_level: int = 0,
        _style_config: StyleConfig | None = None,
    ) -> ValidationResult:
        if data[start_index:] == "-":
            return ValidationResult(True, None)
        regex = r"^-?([1-9][0-9]*|0)(\.0*)?"
        match = re.match(regex, data[start_index:])
        if match:
            return ValidationResult(True, start_index + match.end())
        return ValidationResult(False, None)


class NumberSchemaParser(JsonSchemaParser):
    """A schema parser for number values"""

    def __init__(
        self,
        maximum: float | None,
        minimum: float | None,
        multiple_of: int | None,
        exclusive_maximum: bool,
        exclusive_minimum: bool,
    ):
        # TODO: make it work with float multiples
        self.maximum = maximum
        self.minimum = minimum
        self.multiple_of = multiple_of
        self.exclusive_maximum = exclusive_maximum
        self.exclusive_minimum = exclusive_minimum

    def is_too_much(self, number: float) -> bool:
        """Check if the number is too much for the schema"""
        if self.maximum is None:
            return False
        if self.exclusive_maximum:
            return number >= self.maximum
        return number > self.maximum

    def is_too_little(self, number: float) -> bool:
        """Check if the number is too little for the schema"""
        if self.minimum is None:
            return False
        if self.exclusive_minimum:
            return number <= self.minimum
        return number < self.minimum

    def all_multiples_in_range(self, number: float) -> Iterator[int]:
        """Return all multiples between the maximum and minimum"""
        if self.multiple_of is None:
            return []
        maximum = (
            self.maximum
            or number * (10 * 10 ** int(math.log10(self.multiple_of)))
            + self.multiple_of
        )
        minimum = (
            self.minimum
            or number * (10 * 10 ** int(math.log10(self.multiple_of)))
            - self.multiple_of
        )
        if maximum is None or minimum is None:
            return
        if not minimum < maximum:
            return
        for multiple in range(
            int(minimum / self.multiple_of),
            int(maximum / self.multiple_of) + 1,
        ):
            if multiple * self.multiple_of < minimum:
                continue
            if multiple * self.multiple_of > maximum:
                continue
            yield multiple * self.multiple_of
        return

    def any_multiple_startswith(self, number: int):
        """Check if any multiple starts with the number"""
        if self.maximum is None and self.minimum is None:
            return False
        for multiple in self.all_multiples_in_range(number):
            if str(multiple).startswith(str(number)):
                return True
        return False

    def allowed_negative(self) -> bool:
        """Check whether negative numbers are allowed at all"""
        if self.minimum is None:
            return True
        return self.minimum < 0

    def allowed_positive(self) -> bool:
        """Check whether positive numbers are allowed at all"""
        if self.maximum is None:
            return True
        return self.maximum > 0

    @classmethod
    def load(
        cls, schema: dict, _recursive_parsers: list[Type[JsonSchemaParser]]
    ) -> NumberSchemaParser | None:
        if schema["type"] == "number":
            return cls(
                schema.get("maximum"),
                schema.get("minimum"),
                schema.get("multipleOf"),
                schema.get("exclusiveMaximum", False),
                schema.get("exclusiveMinimum", False),
            )
        return None

    def partial_validate(self, string_segment: str) -> tuple[bool, bool]:
        is_negative = string_segment.startswith("-")
        if not is_negative and not self.allowed_positive():
            return (False, False)
        if is_negative and not self.allowed_negative():
            return (False, False)

        number = float(string_segment)
        valid_full = True

        if self.is_too_much(number):
            if not is_negative:
                return (False, False)
            valid_full = False

        if self.is_too_little(number):
            if is_negative:
                return (False, False)
            valid_full = False

        if self.multiple_of is not None:
            if "." in string_segment:
                return (False, False)
            if int(number) % self.multiple_of:
                valid_full = False
            if not self.any_multiple_startswith(int(number)):
                return (False, False)
        return (True, valid_full)

    def validate(
        self,
        data: str,
        start_index: int = 0,
        _indent_level: int = 0,
        _style_config: StyleConfig | None = None,
    ) -> ValidationResult:
        # I'm going to restrict scientific notation for now because it's a pain
        if data[start_index:] == "-":
            return ValidationResult(self.allowed_negative(), None)
        regex = r"-?([1-9][0-9]*|0)(\.[0-9]*)?"
        match = re.match(regex, data[start_index:])
        if not match:
            return ValidationResult(False, None)
        string_segment = data[start_index : start_index + match.end()]

        valid_partial, valid_full = self.partial_validate(string_segment)
        if not valid_partial:
            return ValidationResult(False, None)
        return ValidationResult(True, start_index + match.end() if valid_full else None)


class StringSchemaParser(JsonSchemaParser):
    """A schema parser for string values"""

    def __init__(
        self, min_length: int | None, max_length: int | None, enum: list[str] | None
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.enum = enum

    @classmethod
    def load(
        cls, schema: dict, _recursive_parsers: list[Type[JsonSchemaParser]]
    ) -> StringSchemaParser | None:
        if schema["type"] == "string":
            return cls(
                schema.get("minLength"), schema.get("maxLength"), schema.get("enum")
            )
        return None

    def valid_length(self, n_chars: int) -> bool:
        if self.min_length is not None and n_chars < self.min_length:
            return False
        if self.max_length is not None and n_chars > self.max_length:
            return False
        return True

    def satisfies_enum_fully(self, data: str) -> bool:
        if self.enum is None:
            return True
        return data in [json.dumps(enum) for enum in self.enum]

    def satisfies_enum_partially(self, data: str) -> bool:
        if self.enum is None:
            return True
        for enum in self.enum:
            if json.dumps(enum).startswith(data):
                return True
        return False

    def validate(
        self,
        data: str,
        start_index: int = 0,
        _indent_level: int = 0,
        _style_config: StyleConfig | None = None,
    ) -> ValidationResult:
        parsed = data[start_index:]
        if not parsed.startswith('"'):
            return ValidationResult(False, None)
        closed = False
        index = 1
        n_chars = 0
        while index < len(parsed):
            if parsed[index] == '"':
                closed = True
                break
            if parsed[index] == "\n":
                return ValidationResult(False, None)
            if parsed[index] == "\\":
                if index + 1 < len(parsed) and parsed[index + 1] == "\n":
                    return ValidationResult(False, None)
                index += 1
            index += 1
            n_chars += 1

        if not closed:
            satisfies_length = (
                self.max_length is None
                or (len(parsed) - (0 if parsed.endswith("\\") else 1))
                <= self.max_length
            )
            satisfies_enum_partially = self.satisfies_enum_partially(parsed)
            return ValidationResult(satisfies_length and satisfies_enum_partially, None)
        if not self.valid_length(n_chars):
            return ValidationResult(False, None)
        if not self.satisfies_enum_fully(parsed[: index + 1]):
            return ValidationResult(False, None)
        return ValidationResult(True, start_index + index + 1)


class AnyOfSchemaParser(JsonSchemaParser):
    """A schema parser for anyOf values"""

    def __init__(self, any_of: list[JsonSchemaParser]):
        self.any_of = any_of

    @classmethod
    def load(
        cls, schema: dict, recursive_parsers: list[Type[JsonSchemaParser]]
    ) -> AnyOfSchemaParser | None:
        if "anyOf" in schema:
            return cls(
                [
                    cls.parser_for(schema, recursive_parsers)
                    for schema in schema["anyOf"]
                ]
            )
        return None

    def validate(
        self,
        data: str,
        start_index: int = 0,
        indent_level: int = 0,
        style_config: StyleConfig | None = None,
    ) -> ValidationResult:
        valid = False
        for parser in self.any_of:
            result = parser.validate(data, start_index, indent_level, style_config)
            if result.valid:
                valid = True
                if result.end_index is not None:
                    return result
        return ValidationResult(valid, None)


class ArraySchemaParser(JsonSchemaParser):
    """A schema parser for array values"""

    def __init__(self, items: JsonSchemaParser):
        self.items = items

    @classmethod
    def load(
        cls, schema: dict, recursive_parsers: list[Type[JsonSchemaParser]]
    ) -> ArraySchemaParser | None:
        if schema["type"] == "array":
            return cls(
                cls.parser_for(schema["items"], recursive_parsers),
            )
        return None

    def skip_indent(
        self, data: str, start_index: int, indent: int | None
    ) -> int | None:
        """Skip indentation in a string"""
        if indent is not None:
            if not (" " * indent).startswith(data[start_index : start_index + indent]):
                return None
            return start_index + indent
        index = start_index
        while index < len(data) and data[index] == " ":
            index += 1
        return index

    def skip_whitespace(
        self,
        data: str,
        start_index: int,
        role: Literal["begin", "after_item", "after_sep"],
        config: StyleConfig,
        indent_level: int,
        is_multiline: bool,
    ) -> int | None:
        """Skip whitespace in a string"""
        item_indent_level = config.indent and indent_level + config.indent
        if role == "begin":
            if is_multiline:
                return self.skip_indent(data, start_index + 1, item_indent_level)
            return start_index
        if role == "after_item":
            if is_multiline and start_index < len(data):
                if data[start_index] == "\n":
                    if (
                        start_index + indent_level + 1 < len(data)
                        and data[start_index + indent_level + 1] != "]"
                    ):
                        return None
                    return self.skip_indent(data, start_index + 1, indent_level)
                if data[start_index] == "]":
                    return None
            if config.force_instant_comma:
                return start_index
            return self.skip_indent(data, start_index, None)
        if role == "after_sep":
            if is_multiline:
                if start_index < len(data) and data[start_index] != "\n":
                    return None
                return self.skip_indent(data, start_index + 1, item_indent_level)
            return self.skip_indent(
                data, start_index, 1 if config.compact_objects else None
            )
        return None

    def validate(
        self,
        data: str,
        start_index: int = 0,
        indent_level: int = 0,
        style_config: StyleConfig | None = None,
    ) -> ValidationResult:
        """Validate an array against a schema"""
        parsed = data[start_index:]
        if not parsed.startswith("["):
            return ValidationResult(False, None)
        if parsed.startswith("[]"):
            return ValidationResult(True, start_index + 2)
        is_multiline = parsed.startswith("[\n")
        if (
            len(parsed) >= 2
            and not is_multiline
            and style_config
            and not style_config.max_inline_array_elements
        ):
            return ValidationResult(False, None)
        index = self.skip_whitespace(
            parsed,
            1,
            "begin",
            style_config or StyleConfig(),
            indent_level,
            is_multiline,
        )
        if index is None:
            return ValidationResult(False, None)

        n_items = 0

        while True:
            # If it ends here
            if index >= len(parsed):
                return ValidationResult(True, None)

            # Get the item
            result = self.items.validate(
                parsed, index, indent_level + (4 if is_multiline else 0), style_config
            )
            if not result.valid:
                return ValidationResult(False, None)
            if result.end_index is None:
                return ValidationResult(True, None)
            index = result.end_index
            n_items += 1

            # Skip whitespace
            index = self.skip_whitespace(
                parsed,
                index,
                "after_item",
                style_config or StyleConfig(),
                indent_level,
                is_multiline,
            )
            if index is None:
                return ValidationResult(False, None)

            # If it ends here
            if index >= len(parsed):
                return ValidationResult(True, None)

            # Check for end of array
            if parsed[index] == "]":
                # If it's multiline, we expect whitespace or a newline before this
                if is_multiline and parsed[index - 1] not in ["\n", " "]:
                    return ValidationResult(False, None)
                return ValidationResult(True, start_index + index + 1)

            # If we got too many items still
            if (
                style_config
                and not is_multiline
                and style_config.max_inline_array_elements is not None
                and style_config.max_inline_array_elements <= n_items
            ):
                return ValidationResult(False, None)

            # Assert comma
            if parsed[index] != ",":
                return ValidationResult(False, None)
            index += 1

            # Skip whitespace
            index = self.skip_whitespace(
                parsed,
                index,
                "after_sep",
                style_config or StyleConfig(),
                indent_level,
                is_multiline,
            )
            if index is None:
                return ValidationResult(False, None)


class ObjectSchemaParser(JsonSchemaParser):
    """A schema parser for object values"""

    def __init__(
        self,
        properties: dict[str, JsonSchemaParser],
        additional_properties: JsonSchemaParser | None,
        required_properties: set[str] | None,
        enforced_order: list[str] | None = None,
    ):
        self.properties = properties
        self.additional_properties = additional_properties
        self.required_properties = required_properties
        self.enforced_order = enforced_order

    @classmethod
    def load(
        cls, schema: dict, recursive_parsers: list[Type[JsonSchemaParser]]
    ) -> ObjectSchemaParser | None:
        if schema["type"] == "object":
            properties = {}
            for property_name, property_schema in schema.get("properties", {}).items():
                properties[property_name] = cls.parser_for(
                    property_schema, recursive_parsers
                )
            additional_properties = None
            if "additionalProperties" in schema and schema["additionalProperties"]:
                additional_properties = cls.parser_for(
                    schema["additionalProperties"], recursive_parsers
                )
            return cls(
                properties,
                additional_properties,
                schema.get("required"),
                schema.get("enforceOrder"),
            )
        return None

    def skip_indent(
        self, data: str, start_index: int, indent: int | None
    ) -> int | None:
        """Skip indentation in a string"""
        if indent is not None:
            if not (" " * indent).startswith(data[start_index : start_index + indent]):
                return None
            return start_index + indent
        index = start_index
        while index < len(data) and data[index] == " ":
            index += 1
        return index

    def skip_whitespace(
        self,
        data: str,
        start_index: int,
        role: Literal[
            "begin", "after_property", "after_colon", "after_item", "after_sep"
        ],
        config: StyleConfig,
        indent_level: int,
        is_multiline: bool,
        needs_to_continue: bool = True,
    ) -> int | None:
        """Skip whitespace in a string"""
        item_indent_level = config.indent and indent_level + config.indent
        if role == "begin":
            if is_multiline:
                return self.skip_indent(data, start_index + 1, item_indent_level)
            return start_index
        if role == "after_property":
            if config.compact_objects:
                return start_index
            return self.skip_indent(data, start_index, None)
        if role == "after_colon":
            if config.compact_objects:
                return self.skip_indent(
                    data, start_index, 1 if config.compact_objects else None
                )
        if role == "after_item":
            if is_multiline and start_index < len(data):
                if data[start_index] == "\n":
                    if needs_to_continue:
                        return start_index
                    if (
                        start_index + indent_level + 1 < len(data)
                        and data[start_index + indent_level + 1] != "}"
                    ):
                        return None
                    return self.skip_indent(data, start_index + 1, indent_level)
                if data[start_index] == "}":
                    return None
            if config.force_instant_comma:
                return start_index
            return self.skip_indent(data, start_index, None)
        if role == "after_sep":
            if is_multiline:
                if start_index < len(data) and data[start_index] != "\n":
                    return None
                return self.skip_indent(data, start_index + 1, item_indent_level)
            return self.skip_indent(
                data, start_index, 1 if config.compact_objects else None
            )
        return None

    def get_parser_for(self, property_name: str) -> JsonSchemaParser | None:
        """Get the parser for a property"""
        if property_name in self.properties:
            return self.properties[property_name]
        if self.additional_properties is not None:
            return self.additional_properties
        return None

    def formulate_enum(
        self, needed_properties: set[str], order: list[str] | None
    ) -> list[str] | None:
        allowed_properties: list[str] | None = (
            list(needed_properties) if self.additional_properties is None else None
        )
        if not order:
            return allowed_properties
        return [order[0]]

    def update_order(self, order: list[str] | None, property_name: str):
        if not order:
            return
        if order[0] != property_name:
            raise ValueError("Order broken")
        del order[0]

    def validate(
        self,
        data: str,
        start_index: int = 0,
        indent_level: int = 0,
        style_config: StyleConfig | None = None,
    ) -> ValidationResult:
        """Validate an object against a schema"""
        needed_properties = set(self.required_properties or self.properties.keys())
        order = self.enforced_order and list(self.enforced_order)

        parsed = data[start_index:]
        if not parsed.startswith("{"):
            return ValidationResult(False, None)
        if parsed.startswith("{}") and not needed_properties:
            return ValidationResult(True, 2)
        is_multiline = parsed.startswith("{\n")
        if (
            len(parsed) >= 2
            and not is_multiline
            and style_config
            and style_config.object_multiline_treshold is not None
            and len(needed_properties) > style_config.object_multiline_treshold
        ):
            return ValidationResult(False, None)
        index = self.skip_whitespace(
            parsed,
            1,
            "begin",
            style_config or StyleConfig(),
            indent_level,
            is_multiline,
        )
        if index is None:
            return ValidationResult(False, None)
        while True:
            # If it ends here
            if index >= len(parsed):
                return ValidationResult(True, None)

            # Get the property name
            result = StringSchemaParser(
                None, None, self.formulate_enum(needed_properties, order)
            ).validate(parsed, index)
            if not result.valid:
                return ValidationResult(False, None)
            if result.end_index is None:
                return ValidationResult(True, None)

            # Extract the property name
            property_name_packed = parsed[index : result.end_index]
            property_name = json.loads(property_name_packed)
            index = result.end_index

            # Skip whitespace
            index = self.skip_whitespace(
                parsed,
                index,
                "after_property",
                style_config or StyleConfig(),
                indent_level,
                is_multiline,
            )
            if index is None:
                return ValidationResult(False, None)

            # If it ends here
            if index >= len(parsed):
                return ValidationResult(True, None)

            # Assert colon
            if parsed[index] != ":":
                return ValidationResult(False, None)
            index += 1

            # Skip whitespace
            index = self.skip_whitespace(
                parsed,
                index,
                "after_colon",
                style_config or StyleConfig(),
                indent_level,
                is_multiline,
            )
            if index is None:
                return ValidationResult(False, None)

            # If it ends here
            if index >= len(parsed):
                return ValidationResult(True, None)

            # Get the parser for the property
            parser = self.get_parser_for(property_name)
            if parser is None:
                return ValidationResult(False, None)

            # Validate the property
            result = parser.validate(
                parsed, index, indent_level + (4 if is_multiline else 0), style_config
            )
            if not result.valid:
                return ValidationResult(False, None)
            if result.end_index is None:
                return ValidationResult(True, None)
            index = result.end_index

            # Update the used properties
            if property_name in needed_properties:
                needed_properties.remove(property_name)
                self.update_order(order, property_name)

            # Skip whitespace
            index = self.skip_whitespace(
                parsed,
                index,
                "after_item",
                style_config or StyleConfig(),
                indent_level,
                is_multiline,
                bool(needed_properties),
            )
            if index is None:
                return ValidationResult(False, None)

            # If it ends here
            if index >= len(parsed):
                return ValidationResult(True, None)

            # Check for end of object
            if parsed[index] == "}":
                # Did we cover all required properties?
                if needed_properties:
                    return ValidationResult(False, None)
                return ValidationResult(True, start_index + index + 1)

            # Assert we can add more properties
            if self.additional_properties is None and not needed_properties:
                return ValidationResult(False, None)

            # Assert comma
            if parsed[index] != ",":
                return ValidationResult(False, None)
            index += 1

            # Skip whitespace
            index = self.skip_whitespace(
                parsed,
                index,
                "after_sep",
                style_config or StyleConfig(),
                indent_level,
                is_multiline,
            )
            if index is None:
                return ValidationResult(False, None)


default_parsers: list[Type[JsonSchemaParser]] = [
    LiteralValueSchemaParser,
    IntegerSchemaParser,
    NumberSchemaParser,
    StringSchemaParser,
    AnyOfSchemaParser,
    ArraySchemaParser,
    ObjectSchemaParser,
]


def parser_for_schema(
    schema: dict, parsers: list[Type[JsonSchemaParser]] | None = None
) -> JsonSchemaParser | None:
    """Get a parser for a schema"""
    _parsers: list[Type[JsonSchemaParser]] = (
        default_parsers if parsers is None else parsers
    )
    for parser in _parsers:
        result = parser.load(schema, _parsers)
        if result is not None:
            return result
    return None
