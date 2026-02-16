import os
import json
import shutil
import pytest

from mcpo.main import build_main_app

try:
    from httpx import AsyncClient, ASGITransport
except ImportError:  # pragma: no cover
    AsyncClient = None  # type: ignore
    ASGITransport = None  # type: ignore


@pytest.mark.asyncio
@pytest.mark.integration
async def test_time_server_new_york_openapi_discovery():
    """Integration test: discover time server tool via OpenAPI and request New York time.

    Skips if 'uvx' (uv tool runner) is not available or httpx AsyncClient missing.
    Mimics OpenWebUI model behavior: fetch sub-app OpenAPI, locate tool, build JSON body, invoke.
    """
    if AsyncClient is None or ASGITransport is None:
        pytest.skip("httpx not installed or AsyncClient API not available")
    if shutil.which("uvx") is None:
        pytest.skip("uvx command not available in PATH; skipping real time server integration test")

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcpo.json")
    if not os.path.exists(config_path):  # safety
        pytest.skip("mcpo.json config not found")

    app = await build_main_app(config_path=config_path, host="127.0.0.1", port=0, api_key="")

    # Use AsyncClient with transport for testing
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # Fetch the time server OpenAPI spec
        # Mounted at /time
        r = await client.get("/time/openapi.json")
        if r.status_code != 200:
            pytest.skip(f"Unable to fetch time server openapi.json (status {r.status_code})")
        spec = r.json()
        paths = spec.get("paths", {})

        target_path = None
        timezone_param_name = None
        # Look for a POST path with a requestBody schema containing a 'timezone' or similar field
        for p, methods in paths.items():
            post = methods.get("post")
            if not post:
                continue
            rb = post.get("requestBody", {})
            content = rb.get("content", {})
            app_json = content.get("application/json", {})
            schema = app_json.get("schema", {})
            props = schema.get("properties", {})
            if not props:
                continue
            for name in props.keys():
                lname = name.lower()
                if "tz" in lname or "zone" in lname or "timezone" in lname:
                    target_path = p
                    timezone_param_name = name
                    break
            if target_path:
                break

        if not target_path or not timezone_param_name:
            pytest.skip("Could not auto-discover timezone parameter in time server OpenAPI spec")

        body = {timezone_param_name: "America/New_York"}
        # POST to /time{target_path}
        resp = await client.post(f"/time{target_path}", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("ok") is True
        # Ensure we got some result payload
        assert "result" in data
        # Basic sanity: result should not be empty
        assert data["result"] not in (None, "")
