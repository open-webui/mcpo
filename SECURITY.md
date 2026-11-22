# Security Policy

mcpo is led and maintained by the same core team as Open WebUI, and it follows the same security process.

## Relationship to the Open WebUI Security Policy

The [Open WebUI Security Policy](https://docs.openwebui.com/security/security-policy/) applies to mcpo in full wherever it is semantically applicable. That includes, and is not limited to, the rules for reporting a vulnerability, the proof-of-concept and remediation requirements, AI disclosure, CVSS accuracy, admin actions being out of scope, default configuration testing, already-fixed reports, self-affecting issues, duplicate and consolidation handling, our position on foreign CNAs and vendor disposition, and responsible disclosure.

Read that policy before filing. Everything in it that can sensibly be applied to mcpo is applied to mcpo. This document adds what is specific to this project, and where the two documents differ, this one governs reports filed here.

## Reporting Channel

Report through [GitHub Security Advisories on this repository](https://github.com/open-webui/mcpo/security/advisories/new). Reports for mcpo code filed against `open-webui/open-webui` are closed there and have to be re-filed here, which costs you time, so file in the right place from the start.

## Supported Versions

Only the latest release is supported. Reproduce your finding against the latest release or against the `dev` branch before filing.

## Threat Model

mcpo is a self-hosted proxy that exposes the tools of MCP servers as an HTTP API. Reports are assessed inside this model:

1. **The operator is trusted.** The command line, the config file, the environment and the choice of MCP servers belong to the operator. Anything that needs control over the operator's settings falls under admin actions and is out of scope.
2. **The API key is the boundary of the HTTP API.** When `--api-key` is set, it guards tool calls, and with `--strict-auth` every endpoint of the API. Whoever holds it can do everything the configured tools can do. Without `--api-key`, the API is open to anyone who can reach it. There are no users and no per-caller permissions.
3. **Tools do what they are built to do.** mcpo relays calls and their content. What a tool can do, and what its descriptions and results contain, belongs to that MCP server. mcpo is responsible for what it does with that content itself. Vulnerabilities in an MCP server go to that server's project.
4. **One instance is one trust boundary.** Everyone who calls the same instance shares its connections and its upstream credentials. Callers who need to be kept apart get separate instances.

## What We Do Want to Hear About

Findings that cross the API key boundary, or that reach beyond what the operator configured, are in scope, including the following.

- Calling a tool without a valid key, or any other bypass of the key check.
- A caller doing more than the configured tools allow.
- A remote MCP server reaching beyond its own tools, into mcpo, the host or other servers' credentials.
- Anything else that affects the operator, the host or a party other than the reporter beyond what the threat model above allows.

When in doubt, file it. A report that lands outside the scope above is closed quickly and without drama, and we would rather read one of those than miss something real.

---

_Last updated on **2026-11-22**._
