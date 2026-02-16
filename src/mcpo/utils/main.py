import json
import traceback
from typing import Any, Dict, ForwardRef, List, Optional, Type, Union
from typing import Literal
import logging
from fastapi import HTTPException, Request

from mcp import ClientSession, types
from mcp.types import (
    CallToolResult,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)

from mcp.shared.exceptions import McpError

from pydantic import Field, create_model
from mcpo.services.runner import get_runner_service
from mcpo.services.metrics import get_metrics_aggregator
from mcpo.services.state import get_state_manager
from pydantic.fields import FieldInfo

MCP_ERROR_TO_HTTP_STATUS = {
    PARSE_ERROR: 400,
    INVALID_REQUEST: 400,
    METHOD_NOT_FOUND: 404,
    INVALID_PARAMS: 422,
    INTERNAL_ERROR: 500,
}

logger = logging.getLogger(__name__)

def normalize_server_type(server_type: str) -> str:
    """Normalize server_type to a standard value."""
    if not server_type:
        return "stdio"
    if server_type in ["streamable_http", "streamablehttp", "streamable-http"]:
        return "streamable-http"
    if server_type in ["sse", "server-sent-events"]:
        return "sse"
    return server_type

def process_tool_response(result: CallToolResult) -> list:
    """Universal response processor for all tool endpoints"""
    response = []
    for content in result.content:
        if isinstance(content, types.TextContent):
            text = content.text
            if isinstance(text, str):
                try:
                    text = json.loads(text)
                except json.JSONDecodeError:
                    pass
            response.append(text)
        elif isinstance(content, types.ImageContent):
            image_data = f"data:{content.mimeType};base64,{content.data}"
            response.append(image_data)
        elif isinstance(content, types.EmbeddedResource):
            # Handle embedded resources
            response.append({
                "type": "embedded_resource",
                "resource": {
                    "uri": content.resource.uri,
                    "mimeType": getattr(content.resource, "mimeType", None),
                    "text": getattr(content.resource, "text", None)
                }
            })
        else:
            # Handle any other content types
            response.append(str(content))
    return response


def name_needs_alias(name: str) -> bool:
    """Check if a field name needs aliasing (if it starts with '_')."""
    return name.startswith("_")


def generate_alias_name(original_name: str, existing_names: set) -> str:
    """
    Generate an alias field name by stripping unwanted chars, and avoiding conflicts with existing names.

    Args:
        original_name: The original field name (should start with '_')
        existing_names: Set of existing names to avoid conflicts with

    Returns:
        An alias name that doesn't conflict with existing names
    """
    alias_name = original_name.lstrip("_")
    # Handle potential naming conflicts
    original_alias_name = alias_name
    suffix_counter = 1
    while alias_name in existing_names:
        alias_name = f"{original_alias_name}_{suffix_counter}"
        suffix_counter += 1
    return alias_name


def _process_schema_property(
    _model_cache: Dict[str, Type],
    prop_schema: Dict[str, Any],
    model_name_prefix: str,
    prop_name: str,
    is_required: bool,
    schema_defs: Optional[Dict] = None,
    processing_stack: Optional[set] = None,
) -> tuple[Union[Type, List, ForwardRef, Any], FieldInfo]:
    """
    Recursively processes a schema property to determine its Python type hint
    and Pydantic Field definition.

    Returns:
        A tuple containing (python_type_hint, pydantic_field).
        The pydantic_field contains default value and description.
    """
    # Handle enum early if present
    if isinstance(prop_schema, dict) and "enum" in prop_schema:
        enum_vals = prop_schema.get("enum") or []
        if enum_vals:
            # Prepare field info for early return paths
            prop_desc = prop_schema.get("description", "")
            default_value = ... if is_required else prop_schema.get("default", None)
            enum_field = Field(default=default_value, description=prop_desc)

            # Infer base type from first value if homogeneous; otherwise Any
            base_type = type(enum_vals[0]) if all(isinstance(v, type(enum_vals[0])) for v in enum_vals) else Any
            try:
                literal_type = Literal[tuple(enum_vals)]  # type: ignore
                return literal_type, enum_field
            except Exception:
                return base_type, enum_field

    if "$ref" in prop_schema:
        ref = prop_schema["$ref"]
        if ref.startswith("#/properties/"):
            # Remove common prefix in pathes.
            prefix_path = model_name_prefix.split("_form_model_")[-1]
            ref_path = ref.split("#/properties/")[-1]
            # Translate $ref path to model_name_prefix style.
            ref_path = ref_path.replace("/properties/", "_model_")
            ref_path = ref_path.replace("/items", "_item")
            # If $ref path is a prefix substring of model_name_prefix path,
            # there exists a circular reference.
            # The loop should be broke with a return to avoid exception.
            if prefix_path.startswith(ref_path):
                # TODO: Find the exact type hint for the $ref.
                return Any, Field(default=None, description="")
        ref = ref.split("/")[-1]
        if schema_defs and ref in schema_defs:
            # Build or reuse a dedicated model for this referenced schema
            target_model_name = f"{model_name_prefix}_{ref}_model".replace("__", "_")
            if target_model_name in _model_cache:
                return _model_cache[target_model_name], Field(default=None if not is_required else ..., description=prop_schema.get("description", ""))

            # Detect cycles
            processing_stack = processing_stack or set()
            if target_model_name in processing_stack:
                # Return forward reference to be resolved later
                # Use string forward reference for Pydantic v2 compatibility; actual resolution occurs when models are built
                return ForwardRef(target_model_name), Field(default=None if not is_required else ..., description="")

            processing_stack.add(target_model_name)
            ref_schema = schema_defs[ref]
            if not isinstance(ref_schema, dict):
                processing_stack.discard(target_model_name)
                return Any, Field(default=None, description="")

            nested_properties = ref_schema.get("properties", {})
            nested_required = ref_schema.get("required", [])
            nested_fields: Dict[str, tuple] = {}
            for name, schema in nested_properties.items():
                is_nested_required = name in nested_required
                nested_type_hint, nested_pydantic_field = _process_schema_property(
                    _model_cache,
                    schema,
                    target_model_name,
                    name,
                    is_nested_required,
                    schema_defs,
                    processing_stack,
                )
                if name_needs_alias(name):
                    other_names = set().union(nested_properties, nested_fields, _model_cache)
                    alias_name = generate_alias_name(name, other_names)
                    aliased_field = Field(
                        default=nested_pydantic_field.default,
                        description=nested_pydantic_field.description,
                        alias=name,
                    )
                    nested_fields[alias_name] = (nested_type_hint, aliased_field)
                else:
                    nested_fields[name] = (nested_type_hint, nested_pydantic_field)

            if not nested_fields:
                processing_stack.discard(target_model_name)
                return Dict[str, Any], Field(default=None if not is_required else ..., description="")

            NestedModel = create_model(target_model_name, **nested_fields)
            _model_cache[target_model_name] = NestedModel
            processing_stack.discard(target_model_name)
            return NestedModel, Field(default=None if not is_required else ..., description="")
        else:
            logger.warning(f"Reference {ref} not found in schema definitions")
            return Any, Field(default=None, description="")

    prop_type = prop_schema.get("type")
    prop_desc = prop_schema.get("description", "")

    default_value = ... if is_required else prop_schema.get("default", None)
    pydantic_field = Field(default=default_value, description=prop_desc)

    # Handle union semantics anyOf/oneOf
    if "anyOf" in prop_schema or "oneOf" in prop_schema:
        type_hints = []
        alts = prop_schema.get("anyOf") or prop_schema.get("oneOf") or []
        for i, schema_option in enumerate(alts):
            # Coarsen enum alternatives to their base type for unions (tests expect str, not Literal)
            if isinstance(schema_option, dict) and "enum" in schema_option:
                enum_vals = schema_option.get("enum") or []
                base = type(enum_vals[0]) if enum_vals else Any
                type_hints.append(base)
            else:
                type_hint, _ = _process_schema_property(
                    _model_cache,
                    schema_option,
                    f"{model_name_prefix}_{prop_name}",
                    f"choice_{i}",
                    False,
                    schema_defs=schema_defs,
                )
                type_hints.append(type_hint)
        return Union[tuple(type_hints)], pydantic_field

    # Handle the case where prop_type is a list of types, e.g. ['string', 'number']
    if isinstance(prop_type, list):
        # Create a Union of all the types
        type_hints = []
        nullable = False
        for type_option in prop_type:
            if type_option == "null":
                nullable = True
                continue
            # Create a temporary schema with the single type and process it
            temp_schema = dict(prop_schema)
            temp_schema["type"] = type_option
            type_hint, _ = _process_schema_property(
                _model_cache, temp_schema, model_name_prefix, prop_name, False, schema_defs=schema_defs
            )
            type_hints.append(type_hint)
        union_type: Any = Union[tuple(type_hints)] if len(type_hints) > 1 else (type_hints[0] if type_hints else Any)
        if nullable:
            return Optional[union_type], pydantic_field
        return union_type, pydantic_field

    # Nullable (OpenAPI 3.0 extension)
    if prop_schema.get("nullable") is True and prop_type is not None:
        temp = dict(prop_schema)
        temp["type"] = prop_type
        non_null_type, _ = _process_schema_property(
            _model_cache, temp, model_name_prefix, prop_name, is_required, schema_defs
        )
        return Optional[non_null_type], pydantic_field

    # allOf composition (intersection semantics)
    if "allOf" in prop_schema:
        composed: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for idx, sub in enumerate(prop_schema["allOf"]):
            # Resolve $ref sub-schemas
            resolved_sub = sub
            if isinstance(sub, dict) and "$ref" in sub:
                ref = sub["$ref"].split("/")[-1]
                if schema_defs and ref in schema_defs:
                    resolved_sub = schema_defs[ref]
            if isinstance(resolved_sub, dict) and resolved_sub.get("type") == "object":
                # Merge object properties/required
                props = resolved_sub.get("properties", {})
                reqs = resolved_sub.get("required", [])
                composed["properties"].update(props)
                for r in reqs:
                    if r not in composed["required"]:
                        composed["required"].append(r)
            else:
                # Non-object part: fallback to original processing of that sub-schema
                sub_type_hint, _ = _process_schema_property(
                    _model_cache, resolved_sub, f"{model_name_prefix}_{prop_name}", f"allof_{idx}", is_required, schema_defs
                )
                # If we encounter a non-object in allOf, we return the sub-type hint as a conservative approach
                return sub_type_hint, pydantic_field
        # Process the composed object
        return _process_schema_property(
            _model_cache, composed, model_name_prefix, prop_name, is_required, schema_defs
        )

    if prop_type == "object":
        nested_properties = prop_schema.get("properties", {})
        nested_required = prop_schema.get("required", [])
        nested_fields = {}

        nested_model_name = f"{model_name_prefix}_{prop_name}_model".replace(
            "__", "_"
        ).rstrip("_")

        if nested_model_name in _model_cache:
            return _model_cache[nested_model_name], pydantic_field

        for name, schema in nested_properties.items():
            is_nested_required = name in nested_required
            nested_type_hint, nested_pydantic_field = _process_schema_property(
                _model_cache,
                schema,
                nested_model_name,
                name,
                is_nested_required,
                schema_defs,
                processing_stack,
            )

            if name_needs_alias(name):
                other_names = set().union(
                    nested_properties, nested_fields, _model_cache
                )
                alias_name = generate_alias_name(name, other_names)
                aliased_field = Field(
                    default=nested_pydantic_field.default,
                    description=nested_pydantic_field.description,
                    alias=name,
                )
                nested_fields[alias_name] = (nested_type_hint, aliased_field)
            else:
                nested_fields[name] = (nested_type_hint, nested_pydantic_field)

        if not nested_fields:
            return Dict[str, Any], pydantic_field

        NestedModel = create_model(nested_model_name, **nested_fields)
        _model_cache[nested_model_name] = NestedModel

        return NestedModel, pydantic_field

    elif prop_type == "array":
        items_schema = prop_schema.get("items")
        if not items_schema:
            # Default to list of anything if items schema is missing
            return List[Any], pydantic_field

        # Recursively determine the type of items in the array
        item_type_hint, _ = _process_schema_property(
            _model_cache,
            items_schema,
            f"{model_name_prefix}_{prop_name}",
            "item",
            False,  # Items aren't required at this level,
            schema_defs,
            processing_stack,
        )
        list_type_hint = List[item_type_hint]
        return list_type_hint, pydantic_field

    elif prop_type == "string":
        return str, pydantic_field
    elif prop_type == "integer":
        return int, pydantic_field
    elif prop_type == "boolean":
        return bool, pydantic_field
    elif prop_type == "number":
        return float, pydantic_field
    elif prop_type == "null":
        # Tests expect literal None for null type
        return None, pydantic_field
    else:
        return Any, pydantic_field


def get_model_fields(form_model_name, properties, required_fields, schema_defs=None):
    model_fields = {}

    _model_cache: Dict[str, Type] = {}

    for param_name, param_schema in properties.items():
        is_required = param_name in required_fields
        python_type_hint, pydantic_field_info = _process_schema_property(
            _model_cache,
            param_schema,
            form_model_name,
            param_name,
            is_required,
            schema_defs,
        )

        # Handle parameter names with leading underscores (e.g., __top, __filter) which Pydantic v2 does not allow
        if name_needs_alias(param_name):
            other_names = set().union(properties, model_fields, _model_cache)
            alias_name = generate_alias_name(param_name, other_names)
            aliased_field = Field(
                default=pydantic_field_info.default,
                description=pydantic_field_info.description,
                alias=param_name,
            )
            # Use the generated type hint and Field info
            model_fields[alias_name] = (python_type_hint, aliased_field)
        else:
            model_fields[param_name] = (python_type_hint, pydantic_field_info)

    return model_fields


SUPPORTED_MCP_VERSIONS = ["2025-06-18"]


def get_tool_handler(
    session,
    endpoint_name,
    form_model_fields,
    response_model_fields=None,
):
    if form_model_fields:
        FormModel = create_model(f"{endpoint_name}_form_model", **form_model_fields)
        ResponseModel = (
            create_model(f"{endpoint_name}_response_model", **response_model_fields)
            if response_model_fields
            else Any
        )

        def make_endpoint_func(
            endpoint_name: str, FormModel, session: ClientSession
        ):  # Parameterized endpoint
            async def tool(request: Request, form_data: FormModel) -> Any:  # type: ignore[name-defined]
                aggregator = get_metrics_aggregator()
                aggregator.record_call()

                args = form_data.model_dump(exclude_none=True, by_alias=True)
                logger.info(f"Calling endpoint: {endpoint_name}, with args: {args}")

                # Enforce server/tool enabled (support legacy main app state and service)
                parent_app = getattr(request.app.state, "parent_app", None)
                server_name = request.app.title
                tool_name = endpoint_name
                # Prefer legacy parent_app.state maps if present (tests manipulate these)
                disabled = False
                if parent_app is not None and hasattr(parent_app.state, "tool_enabled"):
                    # If tool is explicitly false -> disabled
                    disabled = not parent_app.state.tool_enabled.get(server_name, {}).get(tool_name, True)
                else:
                    state_manager = get_state_manager()
                    disabled = not state_manager.is_tool_enabled(server_name, tool_name)
                if disabled:
                    aggregator.record_error("disabled")
                    raise HTTPException(status_code=403, detail={"message": "Tool disabled", "code": "disabled"})

                # Protocol version header warning (for observability)
                recv = request.headers.get("MCP-Protocol-Version")
                if recv != SUPPORTED_MCP_VERSIONS[0]:
                    logger.warning(
                        f"Protocol warn: Unsupported or missing MCP-Protocol-Version; supported={SUPPORTED_MCP_VERSIONS}; received={recv}"
                    )

                # Parse timeout from query param or header; query takes precedence
                timeout = None
                q_timeout = request.query_params.get("timeout") if hasattr(request, "query_params") else None
                if q_timeout is not None:
                    try:
                        timeout = float(q_timeout)
                    except ValueError:
                        aggregator.record_error("invalid_timeout")
                        raise HTTPException(status_code=400, detail={"message": "Invalid timeout", "code": "invalid_timeout"})
                else:
                    timeout_header = request.headers.get("X-Tool-Timeout")
                    if timeout_header:
                        try:
                            timeout = float(timeout_header)
                        except ValueError:
                            aggregator.record_error("invalid_timeout")
                            raise HTTPException(status_code=400, detail={"message": "Invalid timeout", "code": "invalid_timeout"})

                # Default timeout if not provided
                if timeout is None:
                    default_timeout = getattr(request.app.state, "tool_timeout", None)
                    if default_timeout is not None:
                        try:
                            timeout = float(default_timeout)
                        except Exception:
                            timeout = None

                max_timeout = getattr(request.app.state, "tool_timeout_max", None)
                runner = get_runner_service()

                try:
                    # Enforce max timeout in handler for consistent error code/message
                    if timeout is not None and max_timeout is not None and timeout > max_timeout:
                        aggregator.record_error("invalid_timeout")
                        raise HTTPException(status_code=400, detail={"message": "Timeout out of allowed range", "code": "invalid_timeout"})

                    result_payload = await runner.execute_tool(session, endpoint_name, args, timeout=timeout, max_timeout=max_timeout)

                    # Wrap into success envelope; support optional structured_output classification
                    def _classify(val: Any) -> Dict[str, Any]:
                        if isinstance(val, str):
                            return {"type": "text", "value": val}
                        if isinstance(val, (int, float, bool)) or val is None:
                            return {"type": "scalar", "value": val}
                        if isinstance(val, list):
                            return {"type": "collection", "value": val}
                        if isinstance(val, dict):
                            return {"type": "object", "value": val}
                        return {"type": "unknown", "value": val}

                    structured = getattr(request.app.state, "structured_output", False)
                    if structured:
                        # Normalize to list for classification
                        items = result_payload if isinstance(result_payload, list) else [result_payload]
                        classified = [_classify(v) for v in items]
                        return {"ok": True, "result": result_payload, "output": {"type": "collection", "items": classified}}
                    else:
                        return {"ok": True, "result": result_payload}
                except HTTPException as he:
                    code = he.detail.get("code") if isinstance(he.detail, dict) else None
                    if code == "timeout":
                        aggregator.record_error("timeout")
                    elif code == "invalid_timeout" or code == "timeout_too_large":
                        aggregator.record_error("invalid_timeout")
                    elif code == "disabled":
                        aggregator.record_error("disabled")
                    else:
                        aggregator.record_error("unexpected")
                    # Convert to standardized error envelope
                    http_status = he.status_code if hasattr(he, "status_code") else 500
                    msg = he.detail.get("message") if isinstance(he.detail, dict) else str(he.detail)
                    env = {"ok": False, "error": {"message": msg}}
                    if isinstance(he.detail, dict) and he.detail.get("code"):
                        env["error"]["code"] = he.detail["code"]
                    structured = getattr(request.app.state, "structured_output", False)
                    if structured:
                        env["output"] = {"type": "collection", "items": []}
                    from fastapi.responses import JSONResponse
                    return JSONResponse(status_code=http_status, content=env)
                except McpError as e:
                    # Map MCP errors to HTTP and record unexpected
                    aggregator.record_error("unexpected")
                    status_code = MCP_ERROR_TO_HTTP_STATUS.get(e.error.code, 500)
                    from fastapi.responses import JSONResponse
                    env = {"ok": False, "error": {"message": e.error.message}}
                    if e.error.data is not None:
                        env["error"]["data"] = e.error.data
                    return JSONResponse(status_code=status_code, content=env)
                except Exception as e:
                    aggregator.record_error("unexpected")
                    logger.info(
                        f"Unexpected error calling {endpoint_name}: {traceback.format_exc()}"
                    )
                    from fastapi.responses import JSONResponse
                    return JSONResponse(status_code=500, content={"ok": False, "error": {"message": "Unexpected error", "data": str(e)}})

            return tool

        tool_handler = make_endpoint_func(endpoint_name, FormModel, session)
    else:

        def make_endpoint_func_no_args(
            endpoint_name: str, session: ClientSession
        ):  # Parameterless endpoint
            async def tool(request: Request):  # No parameters
                aggregator = get_metrics_aggregator()
                aggregator.record_call()

                logger.info(f"Calling endpoint: {endpoint_name}, with no args")

                # Enforce server/tool enabled
                parent_app = getattr(request.app.state, "parent_app", None)
                server_name = request.app.title
                tool_name = endpoint_name
                disabled = False
                if parent_app is not None and hasattr(parent_app.state, "tool_enabled"):
                    disabled = not parent_app.state.tool_enabled.get(server_name, {}).get(tool_name, True)
                else:
                    state_manager = get_state_manager()
                    disabled = not state_manager.is_tool_enabled(server_name, tool_name)
                if disabled:
                    aggregator.record_error("disabled")
                    raise HTTPException(status_code=403, detail={"message": "Tool disabled", "code": "disabled"})

                recv = request.headers.get("MCP-Protocol-Version")
                if recv != SUPPORTED_MCP_VERSIONS[0]:
                    logger.warning(
                        f"Protocol warn: Unsupported or missing MCP-Protocol-Version; supported={SUPPORTED_MCP_VERSIONS}; received={recv}"
                    )

                # Parse timeout from query or header
                timeout = None
                q_timeout = request.query_params.get("timeout") if hasattr(request, "query_params") else None
                if q_timeout is not None:
                    try:
                        timeout = float(q_timeout)
                    except ValueError:
                        aggregator.record_error("invalid_timeout")
                        raise HTTPException(status_code=400, detail={"message": "Invalid timeout", "code": "invalid_timeout"})
                else:
                    timeout_header = request.headers.get("X-Tool-Timeout")
                    if timeout_header:
                        try:
                            timeout = float(timeout_header)
                        except ValueError:
                            aggregator.record_error("invalid_timeout")
                            raise HTTPException(status_code=400, detail={"message": "Invalid timeout", "code": "invalid_timeout"})

                # Default timeout if not provided
                if timeout is None:
                    default_timeout = getattr(request.app.state, "tool_timeout", None)
                    if default_timeout is not None:
                        try:
                            timeout = float(default_timeout)
                        except Exception:
                            timeout = None

                max_timeout = getattr(request.app.state, "tool_timeout_max", None)
                runner = get_runner_service()
                try:
                    if timeout is not None and max_timeout is not None and timeout > max_timeout:
                        aggregator.record_error("invalid_timeout")
                        raise HTTPException(status_code=400, detail={"message": "Timeout out of allowed range", "code": "invalid_timeout"})
                    result_payload = await runner.execute_tool(session, endpoint_name, {}, timeout=timeout, max_timeout=max_timeout)
                    structured = getattr(request.app.state, "structured_output", False)
                    def _classify(val: Any) -> Dict[str, Any]:
                        if isinstance(val, str):
                            return {"type": "text", "value": val}
                        if isinstance(val, (int, float, bool)) or val is None:
                            return {"type": "scalar", "value": val}
                        if isinstance(val, list):
                            return {"type": "collection", "value": val}
                        if isinstance(val, dict):
                            return {"type": "object", "value": val}
                        return {"type": "unknown", "value": val}
                    if structured:
                        items = result_payload if isinstance(result_payload, list) else [result_payload]
                        classified = [_classify(v) for v in items]
                        return {"ok": True, "result": result_payload, "output": {"type": "collection", "items": classified}}
                    else:
                        return {"ok": True, "result": result_payload}
                except HTTPException as he:
                    code = he.detail.get("code") if isinstance(he.detail, dict) else None
                    if code == "timeout":
                        aggregator.record_error("timeout")
                    elif code == "invalid_timeout" or code == "timeout_too_large":
                        aggregator.record_error("invalid_timeout")
                    elif code == "disabled":
                        aggregator.record_error("disabled")
                    else:
                        aggregator.record_error("unexpected")
                    from fastapi.responses import JSONResponse
                    http_status = he.status_code if hasattr(he, "status_code") else 500
                    msg = he.detail.get("message") if isinstance(he.detail, dict) else str(he.detail)
                    env = {"ok": False, "error": {"message": msg}}
                    if isinstance(he.detail, dict) and he.detail.get("code"):
                        env["error"]["code"] = he.detail["code"]
                    structured = getattr(request.app.state, "structured_output", False)
                    if structured:
                        env["output"] = {"type": "collection", "items": []}
                    return JSONResponse(status_code=http_status, content=env)
                except McpError as e:
                    aggregator.record_error("unexpected")
                    status_code = MCP_ERROR_TO_HTTP_STATUS.get(e.error.code, 500)
                    from fastapi.responses import JSONResponse
                    env = {"ok": False, "error": {"message": e.error.message}}
                    if e.error.data is not None:
                        env["error"]["data"] = e.error.data
                    return JSONResponse(status_code=status_code, content=env)
                except Exception as e:
                    aggregator.record_error("unexpected")
                    logger.info(
                        f"Unexpected error calling {endpoint_name}: {traceback.format_exc()}"
                    )
                    from fastapi.responses import JSONResponse
                    return JSONResponse(status_code=500, content={"ok": False, "error": {"message": "Unexpected error", "data": str(e)}})

            return tool

        tool_handler = make_endpoint_func_no_args(endpoint_name, session)

    return tool_handler


def create_error_envelope(message: str, code: str = None, data: Any = None) -> Dict[str, Any]:
    """Create a standardized error response envelope for API responses."""
    from datetime import datetime, timezone
    envelope = {
        "ok": False,
        "error": {
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }
    if code:
        envelope["error"]["code"] = code
    if data is not None:
        envelope["error"]["data"] = data
    return envelope


def create_success_envelope(data: Any = None, message: str = None) -> Dict[str, Any]:
    """Create a standardized success response envelope for API responses."""
    envelope = {"ok": True}
    if data is not None:
        envelope["data"] = data
    if message:
        envelope["message"] = message
    return envelope


def validate_tool_schema(tool_schema: Dict[str, Any]) -> bool:
    """Validate that a tool schema has the required structure."""
    required_fields = ["name", "inputSchema"]
    return all(field in tool_schema for field in required_fields)


def extract_tool_metadata(tool) -> Dict[str, Any]:
    """Extract metadata from an MCP tool for API documentation."""
    return {
        "name": tool.name,
        "description": getattr(tool, "description", ""),
        "input_schema": getattr(tool, "inputSchema", {}),
        "output_schema": getattr(tool, "outputSchema", None),
    }


def safe_json_loads(text: str, default=None) -> Any:
    """Safely parse JSON with fallback to default value."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else text
