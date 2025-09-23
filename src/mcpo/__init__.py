import sys
import asyncio
import typer
import os
from dotenv import load_dotenv

from typing_extensions import Annotated
from typing import Optional, List

app = typer.Typer()


@app.command(context_settings={"allow_extra_args": True})
def main(
    host: Annotated[
        Optional[str], typer.Option("--host", "-h", help="Host address")
    ] = "0.0.0.0",
    port: Annotated[
        Optional[int], typer.Option("--port", "-p", help="Port number")
    ] = 8000,
    cors_allow_origins: Annotated[
        Optional[List[str]],
        typer.Option("--cors-allow-origins", help="CORS allowed origins"),
    ] = ["*"],
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", "-k", help="API key for authentication"),
    ] = None,
    strict_auth: Annotated[
        Optional[bool],
        typer.Option(
            "--strict-auth", help="API key protects all endpoints and documentation"
        ),
    ] = False,
    env: Annotated[
        Optional[List[str]], typer.Option("--env", "-e", help="Environment variables")
    ] = None,
    env_path: Annotated[
        Optional[str],
        typer.Option("--env-path", help="Path to environment variables file"),
    ] = None,
    server_type: Annotated[
        Optional[str], typer.Option("--type", "--server-type", help="Server type")
    ] = "stdio",
    config_path: Annotated[
        Optional[str], typer.Option("--config", "-c", help="Config file path")
    ] = None,
    name: Annotated[
        Optional[str], typer.Option("--name", "-n", help="Server name")
    ] = None,
    description: Annotated[
        Optional[str], typer.Option("--description", "-d", help="Server description")
    ] = None,
    version: Annotated[
        Optional[str], typer.Option("--version", "-v", help="Server version")
    ] = None,
    ssl_certfile: Annotated[
        Optional[str], typer.Option("--ssl-certfile", "-t", help="SSL certfile")
    ] = None,
    ssl_keyfile: Annotated[
        Optional[str], typer.Option("--ssl-keyfile", "-K", help="SSL keyfile")
    ] = None,
    path_prefix: Annotated[
        Optional[str], typer.Option("--path-prefix", help="URL prefix")
    ] = None,
    headers: Annotated[
        Optional[str], typer.Option("--header", "-H", help="Headers in JSON format")
    ] = None,
    hot_reload: Annotated[
        Optional[bool], typer.Option("--hot-reload", help="Enable hot reload for config file changes")
    ] = False,
):
    server_command = None
    if not config_path:
        # Find the position of "--"
        if "--" not in sys.argv:
            typer.echo("Usage: mcpo --host 0.0.0.0 --port 8000 -- your_mcp_command")
            raise typer.Exit(1)

        idx = sys.argv.index("--")
        server_command: List[str] = sys.argv[idx + 1 :]

        if not server_command:
            typer.echo("Error: You must specify the MCP server command after '--'")
            return

    from mcpo.main import run

    if config_path:
        print("Starting MCP OpenAPI Proxy with config file:", config_path)
    else:
        print(
            f"Starting MCP OpenAPI Proxy on {host}:{port} with command: {' '.join(server_command)}"
        )

    try:
        env_dict = {}
        if env:
            for var in env:
                key, value = var.split("=", 1)
                env_dict[key] = value

        if env_path:
            # Load environment variables from the specified file
            load_dotenv(env_path)
            env_dict.update(dict(os.environ))

        # Set environment variables
        for key, value in env_dict.items():
            os.environ[key] = value
    except Exception as e:
        pass

    # Whatever the prefix is, make sure it starts and ends with a /
    if path_prefix is None:
        # Set default value
        path_prefix = "/"
    # if prefix doesn't end with a /, add it
    if not path_prefix.endswith("/"):
        path_prefix = f"{path_prefix}/"
    # if prefix doesn't start with a /, add it
    if not path_prefix.startswith("/"):
        path_prefix = f"/{path_prefix}"

    # Run your async run function from mcpo.main
    asyncio.run(
        run(
            host,
            port,
            api_key=api_key,
            strict_auth=strict_auth,
            cors_allow_origins=cors_allow_origins,
            server_type=server_type,
            config_path=config_path,
            name=name,
            description=description,
            version=version,
            server_command=server_command,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            path_prefix=path_prefix,
            headers=headers,
            hot_reload=hot_reload,
        )
    )


if __name__ == "__main__":
    app()
