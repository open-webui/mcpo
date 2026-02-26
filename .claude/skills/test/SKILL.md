---
name: test
description: Run the OpenHubUI test suite with pytest
argument-hint: "[test-file-or-pattern]"
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob
---

# Run Tests

Run the OpenHubUI test suite using pytest.

## Default Command

```bash
uv run pytest -v
```

## Targeted Testing

If arguments are provided ($ARGUMENTS), use them to scope the test run:

- **File**: `uv run pytest tests/test_config_models.py -v`
- **Pattern**: `uv run pytest -k "test_hot_reload" -v`
- **Marker**: `uv run pytest -m integration -v`
- **Directory**: `uv run pytest tests/ -v`

## Available Markers

- `asyncio` — async tests (pytest-asyncio)
- `integration` — tests that need external dependencies (MCP servers)

## Test Locations

- `tests/` — Integration and unit tests (~20 test files)
- `src/mcpo/tests/` — In-source tests

## Key Test Files

| File | Coverage |
|------|----------|
| `test_config_models.py` | Pydantic model validation |
| `test_time_server_integration.py` | MCP stdio integration |
| `test_hot_reload.py` | Config reload behavior |
| `test_enable_disable_flow.py` | Tool enable/disable |
| `test_state_persistence.py` | State save/load |
| `test_health_and_timeout.py` | Health checks |
| `test_mcp_tools_integration.py` | MCP tool collection |

## Steps

1. Ensure dev dependencies are installed (`uv sync --dev`)
2. Run the appropriate pytest command
3. Report results — if tests fail, investigate the failures
