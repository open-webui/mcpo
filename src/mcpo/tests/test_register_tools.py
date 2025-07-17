import pytest
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Union

from mcpo.utils.register_tools import _process_schema_property


_model_cache = {}


@pytest.fixture(autouse=True)
def clear_model_cache():
    _model_cache.clear()
    yield
    _model_cache.clear()


def test_process_simple_string_required():
    schema = {"type": "string", "description": "A simple string"}
    expected_type = str
    expected_field = Field(default=..., description="A simple string")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", True
    )
    assert result_type == expected_type
    assert result_field.default == expected_field.default
    assert result_field.description == expected_field.description


def test_process_simple_integer_optional():
    schema = {"type": "integer", "default": 10}
    expected_type = int
    expected_field = Field(default=10, description="")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", False
    )
    assert result_type == expected_type
    assert result_field.default == expected_field.default
    assert result_field.description == expected_field.description


def test_process_simple_boolean_optional_no_default():
    schema = {"type": "boolean"}
    expected_type = bool
    expected_field = Field(
        default=None, description=""
    )  # Default is None if not required and no default specified
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", False
    )
    assert result_type == expected_type
    assert result_field.default == expected_field.default
    assert result_field.description == expected_field.description


def test_process_simple_number():
    schema = {"type": "number"}
    expected_type = float
    expected_field = Field(default=..., description="")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", True
    )
    assert result_type == expected_type
    assert result_field.default == expected_field.default


def test_process_null():
    schema = {"type": "null"}
    expected_type = None
    expected_field = Field(default=..., description="")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", True
    )
    assert result_type == expected_type
    assert result_field.default == expected_field.default


def test_process_unknown_type():
    schema = {"type": "unknown"}
    expected_type = Any
    expected_field = Field(default=..., description="")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", True
    )
    assert result_type == expected_type
    assert result_field.default == expected_field.default


def test_process_array_of_strings():
    schema = {"type": "array", "items": {"type": "string"}}
    expected_type = List[str]
    expected_field = Field(default=..., description="")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", True
    )
    assert result_type == expected_type
    assert result_field.default == expected_field.default


def test_process_array_of_any_missing_items():
    schema = {"type": "array"}  # Missing "items"
    expected_type = List[Any]
    expected_field = Field(default=None, description="")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", False
    )
    assert result_type == expected_type
    assert result_field.default == expected_field.default


def test_process_simple_object():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }
    expected_field = Field(default=..., description="")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", True
    )

    assert result_field.default == expected_field.default
    assert result_field.description == expected_field.description
    assert issubclass(result_type, BaseModel)  # Check if it's a Pydantic model

    # Check fields of the generated model
    model_fields = result_type.model_fields
    assert "name" in model_fields
    assert model_fields["name"].annotation is str
    assert model_fields["name"].is_required()

    assert "age" in model_fields
    assert model_fields["age"].annotation is int
    assert not model_fields["age"].is_required()
    assert model_fields["age"].default is None  # Optional without default


def test_process_nested_object():
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            }
        },
        "required": ["user"],
    }
    expected_field = Field(default=..., description="")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "outer_prop", True
    )

    assert result_field.default == expected_field.default
    assert issubclass(result_type, BaseModel)

    outer_model_fields = result_type.model_fields
    assert "user" in outer_model_fields
    assert outer_model_fields["user"].is_required()

    nested_model_type = outer_model_fields["user"].annotation
    assert issubclass(nested_model_type, BaseModel)

    nested_model_fields = nested_model_type.model_fields
    assert "id" in nested_model_fields
    assert nested_model_fields["id"].annotation is int
    assert nested_model_fields["id"].is_required()


def test_process_array_of_objects():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"item_id": {"type": "string"}},
            "required": ["item_id"],
        },
    }
    expected_field = Field(default=..., description="")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", True
    )

    assert result_field.default == expected_field.default
    assert str(result_type).startswith("typing.List[")  # Check it's a List

    # Get the inner type from List[...]
    item_model_type = result_type.__args__[0]
    assert issubclass(item_model_type, BaseModel)

    item_model_fields = item_model_type.model_fields
    assert "item_id" in item_model_fields
    assert item_model_fields["item_id"].annotation is str
    assert item_model_fields["item_id"].is_required()


def test_process_empty_object():
    schema = {"type": "object", "properties": {}}
    expected_type = Dict[str, Any]  # Should default to Dict[str, Any] if no properties
    expected_field = Field(default=..., description="")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "prop", True
    )
    assert result_type == expected_type
    assert result_field.default == expected_field.default


def test_model_caching():
    schema = {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
    }
    # First call
    result_type1, _ = _process_schema_property(
        _model_cache, schema, "cache_test", "obj1", True
    )
    model_name = "cache_test_obj1_model"
    assert model_name in _model_cache
    assert _model_cache[model_name] == result_type1

    # Second call with same structure but different prefix/prop name (should generate new)
    result_type2, _ = _process_schema_property(
        _model_cache, schema, "cache_test", "obj2", True
    )
    model_name2 = "cache_test_obj2_model"
    assert model_name2 in _model_cache
    assert _model_cache[model_name2] == result_type2
    assert result_type1 != result_type2  # Different models

    # Third call identical to the first (should return cached model)
    result_type3, _ = _process_schema_property(
        _model_cache, schema, "cache_test", "obj1", True
    )
    assert result_type3 == result_type1  # Should be the same cached object
    assert len(_model_cache) == 2  # Only two unique models created


def test_multi_type_property_with_list():
    schema = {
        "type": ["string", "number"],
        "description": "A property with multiple types",
    }
    expected_field = Field(default=..., description="A property with multiple types")
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "multi_type", True
    )

    # Check if the resulting type is a Union
    assert str(result_type).startswith("typing.Union[")

    # Check if both types are in the Union
    assert str in result_type.__args__
    assert float in result_type.__args__

    # Check field properties
    assert result_field.default == expected_field.default
    assert result_field.description == expected_field.description


def test_multi_type_property_with_any_of():
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the function to call",
                    },
                    "arguments": {
                        "type": "string",
                        "description": "The arguments to pass to the function, as a JSON string",
                        "default": "{}",
                    },
                },
                "required": ["name"],
            },
            {
                "type": "object",
                "properties": {
                    "function_id": {
                        "type": "int",
                        "description": "The id of the function to call",
                    },
                },
                "required": ["function_id"],
            },
            {
                "type": "string",
                "enum": ["auto", "none"],
                "description": "Control function calling behavior",
            },
        ],
        "description": "A property with multiple types",
    }
    result_type, result_field = _process_schema_property(
        _model_cache, schema, "test", "multi_type", True
    )

    # Check if the resulting type is a Union
    assert result_type.__origin__ == Union

    # Check if the Union has the correct number of types
    assert len(result_type.__args__) == 3
    assert len(result_type.__args__[0].model_fields) == 2
    assert len(result_type.__args__[1].model_fields) == 1
    assert result_type.__args__[2] is str

    # assert result_field parameter config
    assert result_field.description == "A property with multiple types"


def test_ref_to_parent_node():
    schema = {"$ref": "#/properties/data/properties/children/items"}
    result_type, result_field = _process_schema_property(
        _model_cache,
        schema,
        "generate_fishbone_diagram_form_model_data_model_children_item_model_children",
        "item",
        False,
        {},
    )

    assert result_type == Any
    assert result_field.description == ""
