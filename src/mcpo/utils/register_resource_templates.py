import inspect
from typing import Sequence

from fastapi import FastAPI
from fastapi.params import Depends
from mcp import ClientSession, ReadResourceResult
from pydantic import AnyUrl
from starlette.convertors import Convertor
from starlette.routing import compile_path, replace_params

from mcpo.utils.resource_response import process_resource_response
from mcpo.utils.routes import convert_to_route


async def register_resource_templates(app: FastAPI, session: ClientSession, dependencies: Sequence[Depends]):
    resource_templates_result = await session.list_resource_templates()
    resource_templates = resource_templates_result.resourceTemplates
    for resource_template in resource_templates:
        endpoint_name = resource_template.name
        endpoint_description = resource_template.description
        uri = resource_template.uriTemplate
        # URI templates are not valid URLs, so we need to handle them differently
        parsed_uri = uri
        _,_,convertors = compile_path(parsed_uri)

        app.get(parsed_uri, summary=endpoint_name.replace("_", " ").title(), description=endpoint_description, dependencies=dependencies)(create_resource_handler(session, uri, convertors))


def create_resource_handler(session: ClientSession, url: str, convertors: dict[str, Convertor]):
    async def _handler(**kwargs):
        processed_url, _ = replace_params(url, convertors, kwargs)
        resource_response = await session.read_resource(uri=AnyUrl(url=processed_url))
        return process_resource_response(resource_response)

    param_names = convertors.keys()
    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=str
        )
        for name in param_names
    ]
    _handler.__signature__ = inspect.Signature(parameters)
    return _handler
