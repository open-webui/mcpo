# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.21] - 2026-11-25

### Added

* 📁 **Working Directory for Local Servers**: A local server in the config file can now set "cwd" to the folder it should start in, so servers that read files or scripts by relative path work no matter where mcpo is launched from.
* 🔑 **API Key from an Environment Variable**: The API key can now come from the "MCPO_API_KEY" environment variable, which keeps it out of the process list; "--api-key" still works and wins when both are set.
* 📦 **Embedded Resources, Audio and Resource Links in Tool Results**: Tools that return a file, audio or a resource link now give the model the actual content, with text as text, binary data and audio as data URLs and links as their URL, where mcpo used to return a placeholder or drop the result.
* 📋 **Allowed Values in the OpenAPI Spec**: Tool arguments limited to a fixed set of values now list those values in the spec, so models pick a valid one, and a value outside the list gets a 422 that names the allowed values.
* ⚠️ **Warning for --strict-auth without an API Key**: mcpo now warns at startup when "--strict-auth" is set without an API key, since the flag has no effect on its own.
* 🛡️ **Security Policy**: The repository now has a security policy that explains the threat model and how to report a vulnerability.

### Changed

* 🏠 **Listens on Localhost by Default (Breaking)**: mcpo now listens on 127.0.0.1 unless "--host" is passed, so a bare-metal mcpo reached from another machine or from a Docker container such as Open WebUI now needs "--host 0.0.0.0"; the Docker image still listens on all interfaces.
* 👤 **Docker Image Runs as a Non-Root User (Breaking)**: The image now runs mcpo and its MCP servers as the unprivileged "mcpo" user (uid 1000), OAuth tokens saved to file move from "/root/.mcpo" to "/home/mcpo/.mcpo", existing token volumes need "chown -R 1000:1000", and mounted files must be readable by uid 1000.
* 🤫 **Tool Arguments Kept out of the Default Logs**: Tool calls are still logged, but their arguments, which can contain secrets, now only appear with "--log-level DEBUG".
* 🔒 **Hardened Authentication and OAuth Storage**: API key and Basic auth checks now take the same time whether a guess is close or not, the OAuth token folder is readable only by the user running mcpo, and the OAuth callback page escapes the error text it shows.
* 📌 **MCP SDK Kept Below Version 2**: mcpo now requires an MCP SDK release below 2.0, so new installs no longer pull a major version it does not support yet.
* ⬆️ **Refreshed Dependencies**: All locked dependencies, which the Docker image installs, are on their latest releases within the supported versions, picking up upstream security fixes.

### Fixed

* 🐳 **Docker Image and Newer FastAPI Crashes**: The Docker image builds from the lock file again and starts, and mcpo no longer crashes at startup with newer FastAPI releases, while "--ssl-certfile" and "--ssl-keyfile" keep working.
* 🔁 **Reconnect After an MCP Server Restart**: Tool calls now reconnect when a remote MCP server restarts or expires the session, where they used to fail until mcpo was restarted.
* 🔌 **Tool Calls No Longer Hang on Server Errors**: A 502 or similar failure from a remote MCP server no longer leaves tool calls waiting forever, and calls work again once the server recovers.
* 🧩 **Complex Tool Schemas Load**: Tools whose schemas use tuple-style arrays, references to other parts of the schema, types that refer to themselves or to the root, or true and false in place of a sub-schema now load, where they used to stop the server from loading.
* 🔄 **Hot Reload Reliability**: With "--hot-reload", removed and changed servers stop answering right away, an invalid edit is rejected with the same error as at startup, a failed reload leaves every server running as before, and the old process of a changed or removed server is stopped.
* 🔐 **Pre-Registered OAuth Clients**: A "client_id" and "client_secret" set in the OAuth client metadata are now used, where mcpo always tried dynamic client registration.
* 🧯 **Tool Error Messages**: Errors from a tool now reach the caller with their real message, where they were reported as "Unexpected error".
* 🛑 **Clean Shutdown**: Stopping mcpo no longer logs an error traceback, and it closes every MCP server it started.
* 🪪 **License in Package Metadata**: The package metadata now states the MIT license.

## [0.0.20] - 2026-02-27

### Added

* 🔁 **MCP Connection Manager with Auto-Reconnect**: Introduced MCPConnectionManager to manage the full lifecycle of MCP client sessions and transports. Connections that drop due to a ClosedResourceError are now automatically re-established and retried transparently, eliminating manual restarts for intermittent server disconnects.
* 🔄 **Automatic Retry on Closed Sessions**: Tool calls that encounter a ClosedResourceError will now automatically reconnect to the MCP server and retry the call, providing resilient end-to-end execution without user intervention.

### Changed

* 🧩 **Normalized disabledTools Configuration Key**: The disabled tools setting now accepts both camelCase (disabledTools) and snake_case (disabled_tools) in the config file for backwards compatibility and consistency with other configuration keys.
* 🧹 **Simplified Tool Handler Architecture**: Refactored get_tool_handler to remove the session parameter and nested factory functions in favor of a cleaner design that resolves the session from the request app state at call time, enabling reconnect-aware tool execution.
* 🛡️ **Guarded Lifespan Startup During Hot Reload**: The reload handler now checks for the presence of an async context manager before awaiting sub-app lifespan startup, preventing errors when lifespan context is unavailable.

### Fixed

* 🪛 **Fixed Erroneous Validation Raise in disabledTools**: Removed a misplaced raise statement in validate_server_config that would always throw an error after validating disabled_tools, even when the configuration was correct.
* 🧯 **Graceful Handling of asyncio.CancelledError**: Added explicit CancelledError handling during server creation and lifespan startup to properly roll back routes and log errors instead of crashing silently.

## [0.0.19] - 2025-10-14

### Fixed

* 🔁 **Reverted Client Header Forwarding**: Reverted changes introduced in 0.0.18.

## [0.0.18] - 2025-10-14

### Added

* 🧩 **OAuth Support for Streamable HTTP Servers**: Introduced full OAuth 2.0 support with dynamic client registration for 'streamable_http' MCP servers. This enables secure, standards-compliant authentication flows for both user and service clients.
* 🔧 **Configurable Disabled Tools**: Added 'disabled_tools' option in server configuration, allowing selective disabling of specific tools without modifying code. Useful for managing staged rollouts or limited-access environments.
* 🧠 **Adjustable Log Level at Runtime**: Logging verbosity can now be adjusted dynamically, giving operators finer control over monitoring and debugging noise without requiring restarts.
* 🔁 **Client Header Forwarding**: Added automatic forwarding of client HTTP headers to MCP backends—enabling trace propagation, correlation IDs, and richer observability in distributed environments.
* 🛤️ **Support for Custom Path Prefixes**: Servers can now run under a custom path prefix, allowing flexible deployment within multi-service gateways or reverse proxy environments.
* 📄 **Root Path Documentation**: Added documentation describing behavior and configuration for root path handling in multi-app or nested setups.

### Changed

* 🧰 **Refactored Connection Timeout Defaults**: Connection timeout is now set to 'None' by default to prevent premature disconnects during long-running or streaming tasks.
* 🧹 **Removed Deprecated Python Version Tagging**: Dropped explicit Python version tagging; the project now officially supports Python 3.11 and above without manual tagging overhead.

### Fixed

* 🪛 **Improved Hot Reload Lifespan Tracking**: Fixed an issue where sub-applications created during hot reloads were not properly initialized or cleaned up—ensuring stable lifecycle management during dynamic reconfiguration.
* 🔗 **Symlink Handling in Config Watcher**: Resolved a bug where configuration symlinks were not updating correctly on modification; watcher now tracks and reloads the proper file path automatically.

## [0.0.17] - 2025-07-22

### Added

- 🔄 **Hot Reload Support for Configuration Files**: Added \`--hot-reload\` flag to watch your config file for changes and dynamically reload MCP servers without restarting the application—enabling seamless development workflows and runtime configuration updates.
- 🤫 **HTTP Request Filtering for Cleaner Logs**: Added configurable log filtering to reduce noise from frequent HTTP requests, making debugging and monitoring much clearer in production environments.

### Changed

- ⬆️ **Updated MCP Package to v1.12.1**: Upgraded MCP dependency to resolve compatibility issues with Pydantic and improve overall stability and performance.
- 🔧 **Normalized Streamable HTTP Configuration**: Streamlined configuration syntax for streamable-http servers to align with MCP standards while maintaining backward compatibility.

## [0.0.16] - 2025-07-02

### Added

- 🔁 **Enhanced Endpoint Support for Arbitrary Return Types**: Endpoints can now return any JSON-serializable value—removing limitations on tool outputs and enabling support for more diverse workflows, including advanced data structures or dynamic return formats.
- 🪵 **Improved Log Clarity with Streamlined Print Trace Output**: Internal logging has been upgraded with more structured and interpretable print traces, giving users clearer visibility into backend tool behavior and execution flow—especially helpful when debugging multi-agent sequences or nested toolchains.

### Fixed

- 🔄 **Resolved Infinite Loop Edge Case in Custom Schema Inference**: Fixed a bug where circular references ($ref) caused schema processing to hang or crash in rare custom schema setups—ensuring robust and reliable auto-documentation even for deeply nested models.

## [0.0.15] - 2025-06-06

### Added

- 🔐 **Support for Custom Headers in SSE and Streamable Http MCP Connections**: You can now pass custom HTTP headers (e.g., for authentication tokens or trace IDs) when connecting to SSE or streamable_http servers—enabling seamless integration with remote APIs that require secure or contextual headers.
- 📘 **MCP Server Instructions Exposure**: mcpo now detects and exposes instructions output by MCP tools, bringing descriptive setup guidelines and usage help directly into the OpenAPI schema—so users, UIs, and LLM agents can better understand tool capabilities with zero additional config.
- 🧪 **MCP Exception Stacktrace Printing During Failures**: When a connected MCP server raises an internal error, mcpo now displays the detailed stacktrace from the tool directly in the logs—making debugging on failure dramatically easier for developers and MLops teams working on complex flows.

### Fixed

- 🧽 **Corrected Handling of Underscore Prefix Parameters in Pydantic Modes**: Parameters with leading underscores (e.g. _token) now work correctly without conflict or omission in auto-generated schemas—eliminating validation issues and improving compatibility with tools relying on such parameter naming conventions.

## [0.0.14] - 2025-05-11

### Added

- 🌐 **Streamable HTTP Transport Support**: mcpo now supports MCP servers using the Streamable HTTP transport. This allows for more flexible and robust communication, including session management and resumable streams. Configure via CLI with '--server-type "streamable_http" -- <URL>' or in the config file with 'type: "streamable_http"' and a 'url'.

## [0.0.13] - 2025-05-01

### Added

- 🧪 **Support for Mixed and Union Types (anyOf/nullables)**: mcpo now accurately exposes OpenAPI schemas with anyOf compositions and nullable fields.
- 🧷 **Authentication-Required Docs Access with --strict-auth**: When enabled, the new --strict-auth option restricts access to both the tool endpoints and their interactive documentation pages—ensuring sensitive internal services aren’t inadvertently exposed to unauthenticated users or LLMs.
- 🧬 **Custom Schema Definitions for Complex Models**: Developers can now register custom BaseModel schemas with arbitrary nesting and field variants, allowing precise OpenAPI representations of deeply structured payloads—ensuring crystal-clear docs and compatibility for multi-layered data workflows.
- 🔄 **Smarter Schema Inference Across Data Types**: Schema generation has been enhanced to gracefully handle nested unions, nulls, and fallback types, dramatically improving accuracy in tools using variable output formats or flexible data contracts.

## [0.0.12] - 2025-04-14

### Fixed

- ⏳ **Disabled SSE Read Timeout to Prevent Inactivity Errors**: Resolved an issue where Server-Sent Events (SSE) MCP tools would unexpectedly terminate after 5 minutes of no activity—ensuring durable, always-on connections for real-time workflows like streaming updates, live dashboards, or long-running agents.

## [0.0.11] - 2025-04-12

### Added

- 🌊 **SSE-Based MCP Server Support**: mcpo now supports SSE (Server-Sent Events) MCP servers out of the box—just pass 'mcpo --server-type "sse" -- http://127.0.0.1:8001/sse' when launching or use the standard "url" field in your config for seamless real-time integration with streaming MCP endpoints; see the README for full examples and enhanced workflows with live progress, event pushes, and interactive updates.

## [0.0.10] - 2025-04-10

### Added

- 📦 **Support for --env-path to Load Environment Variables from File**: Use the new --env-path flag to securely pass environment variables via a .env-style file—making it easier than ever to manage secrets and config without cluttering your CLI or exposing sensitive data.
- 🧪 **Enhanced Support for Nested Object and Array Types in OpenAPI Schema**: Tools with complex input/output structures (e.g., JSON payloads with arrays or nested fields) are now correctly interpreted and exposed with accurate OpenAPI documentation—making form-based testing in the UI smoother and integrations far more predictable.
- 🛑 **Smart HTTP Exceptions for Better Debugging**: Clear, structured HTTP error responses are now automatically returned for bad requests or internal tool errors—helping users immediately understand what went wrong without digging through raw traces.

### Fixed

- 🪛 **Fixed --env Flag Behavior for Inline Environment Variables**: Resolved issues where the --env CLI flag silently failed or misbehaved—environment injection is now consistent and reliable whether passed inline with --env or via --env-path.

## [0.0.9] - 2025-04-06

### Added

- 🧭 **Clearer Docs Navigation with Path Awareness**: Optimized the /docs and /[tool]/docs pages to clearly display full endpoint paths when using mcpo --config, making it obvious where each tool is hosted—no more guessing or confusion when running multiple tools under different routes.
- 🛤️ **New --path-prefix Option for Precise Routing Control**: Introduced optional --path-prefix flag allowing you to customize the route prefix for all mounted tools—great for integrating mcpo into existing infrastructures, reverse proxies, or multi-service APIs without route collisions.
- 🐳 **Official Dockerfile for Easy Deployment**: Added a first-party Dockerfile so you can containerize mcpo in seconds—perfect for deploying to production, shipping models with standardized dependencies, and running anywhere with a consistent environment.

## [0.0.8] - 2025-04-03

### Added

- 🔒 **SSL Support via '--ssl-certfile' and '--ssl-keyfile'**: Easily enable HTTPS for your mcpo servers by passing certificate and key files—ideal for securing deployments in production, enabling encrypted communication between clients (e.g. browsers, AI agents) and your MCP tools without external proxies.

## [0.0.7] - 2025-04-03

### Added

- 🖼️ **Image Content Output Support**: mcpo now gracefully handles image outputs from MCP tools—returning them directly as binary image content so users can render or download visuals instantly, unlocking powerful new use cases like dynamic charts, AI art, and diagnostics through any standard HTTP client or browser.

## [0.0.6] - 2025-04-02

### Added

- 🔐 **CLI Auth with --api-key**: Secure your endpoints effortlessly with the new --api-key option, enabling basic API key authentication for instant protection without custom middleware or external auth systems—ideal for public or multi-agent deployments.
- 🌐 **Flexible CORS Access via --cors-allow-origins**: Unlock controlled cross-origin access with the new --cors-allow-origins CLI flag—perfect for integrating mcpo with frontend apps, remote UIs, or cloud dashboards while maintaining CORS security.

### Fixed

- 🧹 **Cleaner Proxy Output**: Dropped None arguments from proxy requests, resulting in reduced clutter and improved interoperability with servers expecting clean inputs—ensuring more reliable downstream performance with MCP tools.
