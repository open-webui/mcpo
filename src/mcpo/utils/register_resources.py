from typing import Sequence

from fastapi import FastAPI
from fastapi.params import Depends
from mcp import ClientSession, ReadResourceResult
from pydantic import AnyUrl

from mcpo.utils.routes import convert_to_route


async def register_resources(app: FastAPI, session: ClientSession, dependencies: Sequence[Depends]):
    resources_result = await session.list_resources()
    resources = resources_result.resources
    for resource in resources:
        endpoint_name = resource.name
        endpoint_description = resource.description
        parsed_uri = convert_to_route(resource.uri)

        app.get(
            parsed_uri,
            summary=endpoint_name.replace("_", " ").title(),
            description=endpoint_description,
            dependencies=dependencies
        )(create_resource_handler(session, str(resource.uri)))


def create_resource_handler(session: ClientSession, url: str):
    async def _handler() -> ReadResourceResult:
        return await session.read_resource(uri=AnyUrl(url=url))

    return _handler