import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Depends
from mcp import ClientSession
from mcp.types import ResourceTemplate, ListResourceTemplatesResult, ReadResourceResult
from starlette.convertors import StringConvertor

from mcpo.utils.register_resource_templates import register_resource_templates, create_resource_handler


@pytest.fixture
def mock_app():
    app = MagicMock(spec=FastAPI)
    app.get = MagicMock(return_value=app)
    return app


@pytest.fixture
def mock_session():
    session = AsyncMock(spec=ClientSession)
    return session


@pytest.fixture
def mock_dependencies():
    return [Depends(lambda: "mock_dependency")]


@pytest.fixture
def mock_resource_templates():
    return [
        ResourceTemplate(
            name="test_resource_1",
            description="Test resource 1 description",
            uriTemplate="resource://{resource_id}",
        ),
        ResourceTemplate(
            name="test_resource_2",
            description="Test resource 2 description",
            uriTemplate="resource://{resource_id}/subresource/{subresource_id}",
        ),
    ]


@pytest.fixture
def mock_convertors():
    return {
        "resource_id": StringConvertor(),
        "subresource_id": StringConvertor(),
    }


@pytest.mark.asyncio
async def test_register_resource_templates(mock_app, mock_session, mock_dependencies, mock_resource_templates):
    mock_session.list_resource_templates.return_value = ListResourceTemplatesResult(resourceTemplates=mock_resource_templates)

    await register_resource_templates(mock_app, mock_session, mock_dependencies)

    assert mock_session.list_resource_templates.called
    assert mock_app.get.call_count == len(mock_resource_templates)

    for i, resource in enumerate(mock_resource_templates):
        call_args = mock_app.get.call_args_list[i][0]
        call_kwargs = mock_app.get.call_args_list[i][1]

        assert call_args[0].startswith("/resource/{resource_id}")

        assert call_kwargs["summary"] == resource.name.replace("_", " ").title()
        assert call_kwargs["description"] == resource.description
        assert call_kwargs["dependencies"] == mock_dependencies


@pytest.mark.asyncio
async def test_create_resource_handler():
    session = AsyncMock(spec=ClientSession)
    url = "resource://{resource_id}"
    convertors = {"resource_id": StringConvertor()}

    expected_result = ReadResourceResult(contents=[])
    session.read_resource.return_value = expected_result

    handler = create_resource_handler(session, url, convertors)

    result = await handler(resource_id="123")

    session.read_resource.assert_called_once()
    call_args = session.read_resource.call_args[1]
    assert "uri" in call_args
    assert str(call_args["uri"]) == "resource://123"
    assert result == []


@pytest.mark.asyncio
async def test_create_resource_handler_multiple_params():
    session = AsyncMock(spec=ClientSession)
    url = "resource://{resource_id}/subresource/{subresource_id}"
    convertors = {
        "resource_id": StringConvertor(),
        "subresource_id": StringConvertor(),
    }

    expected_result = ReadResourceResult(contents=[])
    session.read_resource.return_value = expected_result

    handler = create_resource_handler(session, url, convertors)

    result = await handler(resource_id="123", subresource_id="456")

    session.read_resource.assert_called_once()
    call_args = session.read_resource.call_args[1]
    assert "uri" in call_args
    assert str(call_args["uri"]) == "resource://123/subresource/456"
    assert result == []


@pytest.mark.asyncio
async def test_handler_signature():
    session = MagicMock(spec=ClientSession)
    url = "resource://{resource_id}/subresource/{subresource_id}"
    convertors = {
        "resource_id": StringConvertor(),
        "subresource_id": StringConvertor(),
    }

    handler = create_resource_handler(session, url, convertors)

    import inspect
    sig = inspect.signature(handler)

    assert "resource_id" in sig.parameters
    assert "subresource_id" in sig.parameters

    assert sig.parameters["resource_id"].annotation == str
    assert sig.parameters["subresource_id"].annotation == str
