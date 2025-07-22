from pydantic import AnyUrl
from mcp.types import (
    CallToolResult,
    TextContent,
    ImageContent,
    EmbeddedResource,
    TextResourceContents,
    BlobResourceContents,
    Annotations,
)
from mcpo.utils.main import process_tool_response


class TestProcessToolResponse:
    """Test suite for process_tool_response function"""

    def test_text_content_string(self):
        """Test processing TextContent with plain string"""
        content = TextContent(type="text", text="Hello, World!")
        result = CallToolResult(content=[content], isError=False)

        response = process_tool_response(result)

        assert response == ["Hello, World!"]

    def test_text_content_json_string(self):
        """Test processing TextContent with JSON string"""
        json_text = '{"name": "test", "value": 42}'
        content = TextContent(type="text", text=json_text)
        result = CallToolResult(content=[content], isError=False)

        response = process_tool_response(result)

        assert response == [{"name": "test", "value": 42}]

    def test_text_content_invalid_json(self):
        """Test processing TextContent with invalid JSON string"""
        invalid_json = '{"name": "test", "value":}'
        content = TextContent(type="text", text=invalid_json)
        result = CallToolResult(content=[content], isError=False)

        response = process_tool_response(result)

        assert response == [invalid_json]  # Should return original string

    def test_image_content(self):
        """Test processing ImageContent"""
        content = ImageContent(
            type="image",
            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            mimeType="image/png",
        )
        result = CallToolResult(content=[content], isError=False)

        response = process_tool_response(result)

        expected = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        assert response == [expected]

    def test_embedded_resource_text(self):
        """Test processing EmbeddedResource with TextResourceContents"""
        text_resource = TextResourceContents(
            uri=AnyUrl("file://test.txt"),
            text="This is test content",
            mimeType="text/plain",
        )
        content = EmbeddedResource(type="resource", resource=text_resource)
        result = CallToolResult(content=[content], isError=False)

        response = process_tool_response(result)

        expected = {
            "type": "text_resource",
            "uri": "file://test.txt/",
            "text": "This is test content",
            "mimeType": "text/plain",
        }
        assert response == [expected]

    def test_embedded_resource_blob(self):
        """Test processing EmbeddedResource with BlobResourceContents"""
        blob_resource = BlobResourceContents(
            uri=AnyUrl("file://test.bin"),
            blob="SGVsbG8gV29ybGQ=",  # "Hello World" in base64
            mimeType="application/octet-stream",
        )
        content = EmbeddedResource(type="resource", resource=blob_resource)
        result = CallToolResult(content=[content], isError=False)

        response = process_tool_response(result)

        expected = {
            "type": "blob_resource",
            "uri": "file://test.bin/",
            "data": "data:application/octet-stream;base64,SGVsbG8gV29ybGQ=",
            "mimeType": "application/octet-stream",
        }
        assert response == [expected]

    def test_embedded_resource_blob_no_mimetype(self):
        """Test processing EmbeddedResource with BlobResourceContents without mimeType"""
        blob_resource = BlobResourceContents(
            uri=AnyUrl("file://test.bin"), blob="SGVsbG8gV29ybGQ=", mimeType=None
        )
        content = EmbeddedResource(type="resource", resource=blob_resource)
        result = CallToolResult(content=[content], isError=False)

        response = process_tool_response(result)

        expected = {
            "type": "blob_resource",
            "uri": "file://test.bin/",
            "data": "data:application/octet-stream;base64,SGVsbG8gV29ybGQ=",
            "mimeType": None,
        }
        assert response == [expected]

    def test_embedded_resource_with_annotations(self):
        """Test processing EmbeddedResource with annotations"""
        text_resource = TextResourceContents(
            uri=AnyUrl("file://test.txt"), text="Test content", mimeType="text/plain"
        )

        # Create real annotations
        annotations = Annotations(priority=0.8)

        content = EmbeddedResource(
            type="resource", resource=text_resource, annotations=annotations
        )
        result = CallToolResult(content=[content], isError=False)

        response = process_tool_response(result)

        expected = {
            "type": "text_resource",
            "uri": "file://test.txt/",
            "text": "Test content",
            "mimeType": "text/plain",
            "annotations": {"audience": None, "priority": 0.8},
        }
        assert response == [expected]

    def test_embedded_resource_with_meta(self):
        """Test processing EmbeddedResource with metadata"""
        text_resource = TextResourceContents(
            uri=AnyUrl("file://test.txt"), text="Test content", mimeType="text/plain"
        )
        content = EmbeddedResource(
            type="resource",
            resource=text_resource,
            _meta={"source": "test", "timestamp": "2024-01-01"},
        )
        result = CallToolResult(content=[content], isError=False)

        response = process_tool_response(result)

        expected = {
            "type": "text_resource",
            "uri": "file://test.txt/",
            "text": "Test content",
            "mimeType": "text/plain",
            "meta": {"source": "test", "timestamp": "2024-01-01"},
        }
        assert response == [expected]

    def test_multiple_content_types(self):
        """Test processing multiple different content types in one result"""
        text_content = TextContent(type="text", text="Hello")
        image_content = ImageContent(
            type="image", data="dGVzdA==", mimeType="image/png"
        )
        text_resource = TextResourceContents(
            uri=AnyUrl("file://test.txt"),
            text="Resource content",
            mimeType="text/plain",
        )
        embedded_content = EmbeddedResource(type="resource", resource=text_resource)

        result = CallToolResult(
            content=[text_content, image_content, embedded_content], isError=False
        )

        response = process_tool_response(result)

        expected = [
            "Hello",
            "data:image/png;base64,dGVzdA==",
            {
                "type": "text_resource",
                "uri": "file://test.txt/",
                "text": "Resource content",
                "mimeType": "text/plain",
            },
        ]
        assert response == expected

    def test_empty_content_list(self):
        """Test processing empty content list"""
        result = CallToolResult(content=[], isError=False)

        response = process_tool_response(result)

        assert response == []
