# Dreamtune tools through OpenAPI

Use MCPO to expose [Dreamtune](https://dreamtunelabs.com)'s MCP tools to OpenAPI clients, including Open WebUI. The published stdio server provides model requests, tool runs, GPU compute and run inspection. MCPO generates the HTTP routes and schemas from its MCP tool definitions.

## Start the proxy

Install Python 3.11+, [uv](https://docs.astral.sh/uv/) and Node.js **22.20+**, including `npx`. From this repository's root:

```sh
uv sync --frozen
export DREAMTUNE_API_KEY="YOUR_DREAMTUNE_WORKSPACE_KEY"
export MCPO_API_KEY="YOUR_SEPARATE_PROXY_KEY"
uv run mcpo --host 127.0.0.1 --port 8000 --strict-auth \
  --api-key "$MCPO_API_KEY" --config examples/dreamtune.json
```

PowerShell:

```powershell
uv sync --frozen
$env:DREAMTUNE_API_KEY = "YOUR_DREAMTUNE_WORKSPACE_KEY"
$env:MCPO_API_KEY = "YOUR_SEPARATE_PROXY_KEY"
uv run mcpo --host 127.0.0.1 --port 8000 --strict-auth --api-key $env:MCPO_API_KEY --config examples/dreamtune.json
```

MCPO passes the parent environment to the stdio process. The example therefore inherits `DREAMTUNE_API_KEY`; it does not store your key in JSON or depend on variable substitution in the config loader. The configuration pins `@dreamtune/sdk@0.1.2` and supplies the production API origin and request deadline explicitly.

The proxy key authenticates your OpenAPI client to MCPO. The Dreamtune workspace key authenticates the child process to Dreamtune; these are separate credentials. `--strict-auth` protects documentation and schema endpoints as well as tool routes.

## Inspect the generated tools

```sh
curl --fail -H "Authorization: Bearer $MCPO_API_KEY" \
  http://127.0.0.1:8000/dreamtune/openapi.json
```

This fetches generated tool definitions without invoking a tool or starting paid work. The schema contains routes such as `/list_tools`, `/create_model_run` and `/compute_catalog`, served under the `/dreamtune` mount. Interactive documentation is at `/dreamtune/docs`; with strict authentication, use a client that sends your proxy credential when opening protected documentation.

For [Open WebUI setup](https://docs.openwebui.com/openapi-servers/open-webui/), add the reachable MCPO server URL with the `/dreamtune` suffix and the proxy key as its Bearer credential. `http://127.0.0.1:8000/dreamtune` works when the client runs on the same host; a containerized or remote client needs a reachable host address and appropriate network configuration.

## Calling tools

Use the generated request schema for each operation and the [Dreamtune MCP tool reference](https://docs.dreamtunelabs.com/reference/mcp-tools). Calls use the workspace and scopes of `DREAMTUNE_API_KEY`. Model requests, paid tool runs and compute jobs can incur charges; compute can continue billing until stopped. Persist an idempotency key for operations that require one and reuse it only for identical intended work.

See [local MCP setup](https://docs.dreamtunelabs.com/connect/local-mcp) for optional GPU access configuration. No optional compute state directory is needed to generate the OpenAPI schema.
