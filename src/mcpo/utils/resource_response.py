import json
import logging

from mcp import ReadResourceResult, types

logger = logging.getLogger(__name__)


def process_resource_response(response: ReadResourceResult) -> list:
    """Universal response processor for all resource endpoints, which matches the behaviour of tool handling."""
    response_list = []
    for content in response.contents:
        if isinstance(content, types.TextResourceContents):
            text = content.text
            if isinstance(text, str):
                try:
                    text = json.loads(text)
                except json.JSONDecodeError:
                    logger.info("There was an error decoding the JSON response while processing resource response.")
                    pass
            response_list.append(text)
        elif isinstance(content, types.BlobResourceContents):
            blob_data = f"data:{content.mimeType or 'application/octet-stream'};base64,{content.blob}"
            response_list.append(blob_data)

    if not response_list:
        return response_list
    return response_list if len(response_list) > 1 else response_list[0]
