from mcp import ReadResourceResult, types
from pydantic import AnyUrl

from mcpo.utils.resource_response import process_resource_response


def test_process_resource_response_with_text_content():
    text_content = types.TextResourceContents(
        uri=AnyUrl(url="example://resource"),
        text="Hello, world!",
        mimeType="text/plain"
    )
    response = ReadResourceResult(contents=[text_content])
    
    result = process_resource_response(response)
    
    assert result == ["Hello, world!"]


def test_process_resource_response_with_blob_content():
    blob_content = types.BlobResourceContents(
        uri=AnyUrl(url="example://resource"),
        blob="SGVsbG8sIHdvcmxkIQ==",
        mimeType="application/octet-stream"
    )
    response = ReadResourceResult(contents=[blob_content])
    
    result = process_resource_response(response)
    
    assert result == ["data:application/octet-stream;base64,SGVsbG8sIHdvcmxkIQ=="]


def test_process_resource_response_with_blob_content_no_mime_type():
    blob_content = types.BlobResourceContents(
        uri=AnyUrl(url="example://resource"),
        blob="SGVsbG8sIHdvcmxkIQ==",
        mimeType=None
    )
    response = ReadResourceResult(contents=[blob_content])
    
    result = process_resource_response(response)
    
    assert result == ["data:application/octet-stream;base64,SGVsbG8sIHdvcmxkIQ=="]


def test_process_resource_response_with_multiple_contents():
    text_content = types.TextResourceContents(
        uri=AnyUrl(url="example://resource/text"),
        text="Hello, world!",
        mimeType="text/plain"
    )
    blob_content = types.BlobResourceContents(
        uri=AnyUrl(url="example://resource/blob"),
        blob="SGVsbG8sIHdvcmxkIQ==",
        mimeType="application/octet-stream"
    )
    response = ReadResourceResult(contents=[text_content, blob_content])
    
    result = process_resource_response(response)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "Hello, world!"
    assert result[1] == "data:application/octet-stream;base64,SGVsbG8sIHdvcmxkIQ=="


def test_process_resource_response_with_empty_contents():
    response = ReadResourceResult(contents=[])
    
    result = process_resource_response(response)
    
    assert isinstance(result, list)
    assert len(result) == 0


def test_process_resource_response_with_json_decode_error():
    invalid_json = '{"key": "value", "number": 42'
    text_content = types.TextResourceContents(
        uri=AnyUrl(url="example://resource"),
        text=invalid_json,
        mimeType="application/json"
    )
    response = ReadResourceResult(contents=[text_content])
    
    result = process_resource_response(response)
    
    assert result == [invalid_json]