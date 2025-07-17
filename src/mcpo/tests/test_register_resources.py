import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, Depends
from mcp import ClientSession
from mcp.types import Resource, ListResourcesResult, ReadResourceResult
from pydantic import AnyUrl

from mcpo.utils.register_resources import register_resources, create_resource_handler


@pytest.fixture
def mock_app():
    """Fixture for a mock FastAPI app."""
    app = MagicMock(spec=FastAPI)
    app.get = MagicMock(return_value=app)
    return app


@pytest.fixture
def mock_session():
    """Fixture for a mock ClientSession."""
    session = AsyncMock(spec=ClientSession)
    return session


@pytest.fixture
def mock_dependencies():
    """Fixture for mock dependencies."""
    return [Depends(lambda: "mock_dependency")]


@pytest.fixture
def mock_resources():
    """Fixture for mock resources."""
    return [
        Resource(
            name="test_resource_1",
            description="Test resource 1 description",
            uri=AnyUrl(url="example://resource1"),
        ),
        Resource(
            name="test_resource_2",
            description="Test resource 2 description",
            uri=AnyUrl(url="example://resource2"),
        ),
    ]


@pytest.mark.anyio
async def test_register_resources(mock_app, mock_session, mock_dependencies, mock_resources):
    """Test that register_resources registers handlers for all resources."""
    # Setup
    mock_session.list_resources.return_value = ListResourcesResult(resources=mock_resources)

    # Execute
    await register_resources(mock_app, mock_session, mock_dependencies)

    # Verify
    assert mock_session.list_resources.called
    assert mock_app.get.call_count == len(mock_resources)

    # Check that each resource was registered with the correct endpoint
    for i, resource in enumerate(mock_resources):
        call_args = mock_app.get.call_args_list[i][0]
        call_kwargs = mock_app.get.call_args_list[i][1]

        # Check that the path is correctly formatted
        assert call_args[0] == f"/example/resource{i+1}"

        # Check that the summary and description are set correctly
        assert call_kwargs["summary"] == resource.name.replace("_", " ").title()
        assert call_kwargs["description"] == resource.description
        assert call_kwargs["dependencies"] == mock_dependencies


@pytest.mark.anyio
async def test_create_resource_handler():
    """Test that create_resource_handler creates a handler that calls session.read_resource with correct arguments."""
    # Setup
    session = AsyncMock(spec=ClientSession)
    url = "example://resource"

    expected_result = ReadResourceResult(contents=[])
    session.read_resource.return_value = expected_result

    # Create handler
    handler = create_resource_handler(session, url)

    # Execute handler
    result = await handler()

    # Verify
    session.read_resource.assert_called_once()
    # Check that the URL was processed correctly
    call_args = session.read_resource.call_args[1]
    assert "uri" in call_args
    assert str(call_args["uri"]) == url
    assert result == expected_result


@pytest.mark.anyio
async def test_register_resources_with_empty_resources():
    """Test that register_resources handles empty resources list."""
    # Setup
    app = MagicMock(spec=FastAPI)
    session = AsyncMock(spec=ClientSession)
    dependencies = [Depends(lambda: "mock_dependency")]

    session.list_resources.return_value = ListResourcesResult(resources=[])

    # Execute
    await register_resources(app, session, dependencies)

    # Verify
    assert session.list_resources.called
    assert app.get.call_count == 0
