# OpenCode + Elektron Integration Strategy for OpenHubUI

## Purpose

This document captures a detailed product and engineering strategy for evolving OpenHubUI into a desktop-first, agent-capable environment by integrating:

- **OpenWebUI** as the primary conversational and user-facing interface
- **OpenCode** as the local coding-agent runtime and provider compatibility layer
- **OpenHubUI (MCPO)** as the MCP aggregation, policy, and control plane
- **Electron ("Elektron")** as the desktop shell that unifies all of the above into one app surface

The goal is to reduce server rigidity, increase local capability, improve agentic workflows, and provide a user experience closer to Claude-style coworking while preserving OpenWebUI familiarity.

---

## Problem Statement

### Current friction

OpenWebUI is highly useful but can feel rigid in server-hosted configurations:

- Limited direct local system affordances
- Weakly integrated local filesystem/process permissions model
- Friction around practical agentic tasks that need local execution context
- Inconsistent behavior across providers and tool backends
- Fragmented setup when users need MCP + local tools + chat UI in one place

### Product opportunity

A desktop application can provide:

- Local-first permissions and sandboxing
- Consistent access to files/folders for agentic workflows
- Unified model/provider compatibility layer
- Integrated MCP orchestration and observability
- Single application surface for chat + tooling + runtime controls

---

## Product Vision

### What we are building

A desktop OpenHubUI app that:

1. Embeds a Chromium-based browser pointed at OpenWebUI
2. Runs OpenCode server locally for robust agent runtime
3. Runs OpenHubUI locally for MCP aggregation and admin/policy controls
4. Provides a secure brokered capability model for filesystem/process access
5. Feels like one cohesive product despite multiple internal subsystems

### User-facing outcome

Users launch one app and get:

- Familiar OpenWebUI interface
- Reliable local agent execution
- Configurable, discoverable MCP tool access
- Granular permission prompts and policy controls
- Better compatibility across providers/models

---

## Strategic Positioning

### Core value boundaries

- **OpenWebUI core value:** chat UX, prompt interactions, end-user familiarity
- **OpenCode core value:** coding-agent runtime, provider breadth, session/tool API
- **OpenHubUI core value:** MCP aggregation/proxy, policy enforcement, admin plane
- **Electron core value:** desktop packaging, process orchestration, secure IPC/UI shell

### Why not replace everything with one component

A full replacement approach is high-risk and unnecessary. The better path is compositional:

- Keep each subsystem in its strength area
- Integrate over stable APIs
- Avoid deep forks early
- Preserve independent upgradeability where possible

---

## High-Level Architecture

###[A] Desktop Shell Layer (Electron)

Responsibilities:

- Bootstraps and monitors local sidecar services
- Embeds Chromium view for OpenWebUI URL
- Exposes secure IPC to native capabilities via broker
- Handles app lifecycle, logging, crash recovery, updates

Key constraints:

- No direct unrestricted renderer access to OS primitives
- Strict preload and IPC allowlists
- Service readiness and health-gated UI start

###[B] UI Layer (OpenWebUI in embedded browser)

Responsibilities:

- Main user interaction surface
- Conversations, model usage, tool prompts, rendering
- Optional OpenWebUI plugin/extension points for desktop-enhanced controls

###[C] Agent Runtime Layer (OpenCode)

Responsibilities:

- Session lifecycle, tool orchestration, event streams
- Provider/model compatibility breadth
- Programmatic API via server endpoints/SDK

Notes:

- OpenCode should be consumed via published server and SDK contracts first
- Avoid immediate deep internal coupling or source forking

###[D] MCP & Control Plane Layer (OpenHubUI / MCPO)

Responsibilities:

- Aggregate MCP servers (stdio, SSE, streamable HTTP)
- Tool enable/disable and server-level policy controls
- Configuration, state persistence, operational APIs
- Potential policy enforcement layer for tool exposure

###[E] Capability Broker Layer (Desktop security boundary)

Responsibilities:

- Mediate all filesystem/process/network-sensitive actions
- Enforce policy decisions (allow/deny/prompt/sandbox)
- Emit audit trails and permission histories
- Offer constrained capability tokens/scopes to runtimes

---

## Data and Request Flow (Target)

1. User interacts with OpenWebUI in embedded browser.
2. OpenWebUI invokes tool/agent capability through integrated endpoint(s).
3. Requests route to:
   - OpenCode for agent execution/session workflows
   - OpenHubUI for MCP tool aggregation and policy-governed exposure
4. Any local privileged operation is brokered via desktop permission layer.
5. Results and events stream back to UI with traceable audit metadata.

---

## Security and Trust Model

### Security goals

- Minimize privilege by default
- Keep renderer untrusted
- Prevent silent local privilege escalation
- Provide explicit user consent and revocation
- Preserve auditability for sensitive actions

### Trust boundaries

1. **Renderer boundary:** embedded OpenWebUI cannot directly access filesystem/process APIs.
2. **IPC boundary:** only approved IPC channels with strict schema validation.
3. **Runtime boundary:** OpenCode/OpenHubUI sidecars get scoped capabilities, not blanket host access.
4. **Tool boundary:** MCP tools are treated as potentially high-risk and policy-gated.

### Permission model

Permission should be:

- **Scope-aware:** per workspace, per directory subtree, per action type
- **Time-aware:** one-time, session, persistent grants
- **Principal-aware:** which subsystem/tool requested the action
- **Revocable:** users can view and revoke grants from control panel

Permission categories:

- Read file/directory
- Write/modify file
- Execute shell command
- Launch subprocess
- Network access (outbound)
- Credential/secret access

### Hardening baseline

- Disable Node integration in renderer
- Enable context isolation
- Use strict CSP for embedded/local pages
- Validate all IPC payloads with explicit schemas
- Signed binaries and secure update channels
- No plaintext secret logging

---

## Capability and Sandbox Design

### Execution modes

1. **Safe mode (default):**
   - Read-only project context
   - No arbitrary shell execution
   - Restricted network
2. **Balanced mode:**
   - Controlled edits and command execution with prompts
3. **Power mode (explicit opt-in):**
   - Expanded capability scope with clear warnings and audit tracking

### Workspace-scoped isolation

- Each project gets a scoped policy profile
- Per-workspace state includes:
  - Allowed directories
  - Allowed command classes
  - Allowed MCP servers/tools
  - Preferred model/provider settings

### Tool-level policies

- Global enable/disable by tool name pattern
- Per-agent tool policies
- Per-server MCP policies
- Deny-by-default option for unknown tools in strict environments

---

## Provider and Model Compatibility Strategy

### Current insight

OpenCode already solves a major compatibility burden by supporting broad provider coverage and custom OpenAI-compatible endpoints.

### Proposed strategy

- Use OpenCode as primary compatibility engine for agent/runtime model invocation.
- Reduce bespoke provider logic in OpenHubUI over time.
- Keep OpenHubUI focused on MCP and control-plane concerns.

### Migration principle

**De-duplicate provider logic, not core product identity.**

Do not remove OpenHubUI’s unique value (MCP + policy + ops). Remove custom provider complexity where OpenCode already provides stable primitives.

---

## Integration Approach Options

### Option 1: Thin integration (recommended first)

- OpenHubUI remains primary app backend
- OpenCode runs as sidecar for agent/runtime endpoints
- OpenWebUI embedded in Electron points to local OpenHubUI/OpenWebUI stack

Pros:

- Lower migration risk
- Faster MVP
- Keeps existing architecture mostly intact

Cons:

- Two backend runtimes to orchestrate
- More integration glue initially

### Option 2: Runtime-first integration

- OpenCode handles most chat/agent runtime flows
- OpenHubUI increasingly focuses on MCP aggregation + policy

Pros:

- Greater reduction of custom provider/runtime code
- Better long-term maintainability potential

Cons:

- Requires deliberate API contract mapping
- Higher refactor and regression risk

### Option 3: Full replacement (not recommended early)

- Attempt to fully replace OpenHubUI backend responsibilities with OpenCode internals

Risks:

- High coupling and fork burden
- Loss of control-plane distinctions
- Large migration blast radius

---

## MVP Definition

### MVP objective

Ship a desktop app that feels unified and practical for local agentic work with strong permission controls.

### MVP scope (must-have)

1. Electron app boots and supervises local services:
   - OpenWebUI target
   - OpenHubUI service
   - OpenCode service
2. Embedded browser loads OpenWebUI reliably.
3. Basic cross-service routing for chat/tool/agent execution.
4. Permission broker for file read/write + shell execute prompts.
5. Settings panel for:
   - Workspace path
   - Provider/model defaults
   - MCP server toggles
   - Permission history
6. Health dashboard for all sidecar services.

### MVP explicitly out-of-scope

- Full enterprise multi-tenant auth layer
- Cloud-synced policy profiles
- Deep custom visual theming and non-essential UX polish
- Complete replacement of all existing backend flows

---

## Phased Delivery Plan

### Phase 0 — Architecture and contracts

- Define API contracts between OpenWebUI bridge and sidecars
- Define permission IPC schema and audit event format
- Finalize process supervision and health checks

Deliverables:

- Architecture spec
- Threat model draft
- Interface contracts doc

### Phase 1 — Desktop shell + sidecar orchestration

- Build Electron process manager
- Start/stop/restart sidecars with health probes
- Add minimal diagnostics/log viewer

Deliverables:

- Booting desktop shell
- Sidecar lifecycle manager

### Phase 2 — UI embedding + routing bridge

- Embed OpenWebUI URL in Chromium view
- Implement request routing bridge to OpenCode/OpenHubUI
- Validate core chat/tool round-trips

Deliverables:

- Single-surface usable app
- Basic agentic workflow pass

### Phase 3 — Permission broker + sandbox profiles

- Implement permission prompts and persistence
- Add scope-aware grants and revocation
- Enforce broker path for all privileged operations

Deliverables:

- Working policy UX
- Auditable sensitive operation trail

### Phase 4 — MCP + agent coherence

- Unify tool visibility semantics across OpenCode/OpenHubUI
- Harden disable/deny behavior and edge-case handling
- Improve observability for tool invocation paths

Deliverables:

- Stable tool policy behavior
- Improved debuggability

### Phase 5 — Packaging and production readiness

- Installer flows (Windows first)
- Auto-update channel with signing
- Crash reporting and reliability hardening

Deliverables:

- Production-ready desktop distribution candidate

---

## Technical Risks and Mitigations

### Risk: Cross-runtime complexity

Problem:

- Python + Node/Bun + Electron orchestration can become fragile.

Mitigation:

- Explicit service supervisor with backoff/restart policies
- Strict health/readiness contracts
- Structured logs with correlation IDs

### Risk: Inconsistent tool policy enforcement

Problem:

- Multiple tool paths can bypass intended policy.

Mitigation:

- Central policy decision point in broker/control plane
- Contract tests for allow/deny consistency
- Regression tests around remount/reload paths

### Risk: Security regression via renderer/IPC

Problem:

- Electron misconfiguration can expose local system.

Mitigation:

- Secure Electron defaults and audited IPC channels
- Payload schema validation and explicit deny for unknown channels
- Security review checklist before release

### Risk: Upgrade friction with upstream OpenCode changes

Problem:

- Tight coupling to internals may break during upgrades.

Mitigation:

- Integrate via server/SDK APIs only
- Version pinning and compatibility testing matrix
- Avoid heavy source forks unless necessary

---

## Operational Observability

### Minimum telemetry (local-first, privacy aware)

- Service lifecycle events (start/stop/crash)
- Permission prompts and outcomes
- Tool invocation traces (metadata only by default)
- Policy denials and reasons
- Session and runtime health snapshots

### Debug surfaces

- Unified local log viewer in desktop app
- Service health panel with readiness indicators
- Exportable diagnostics bundle for support

---

## Suggested API and Contract Principles

1. **Stable contracts over internal coupling**
   - Treat OpenCode and OpenHubUI as API products.
2. **Versioned interfaces**
   - Include version headers or schema version fields.
3. **Typed envelopes**
   - Standardize success/error envelope format for brokered requests.
4. **Audit correlation IDs**
   - Every sensitive operation has traceable ID across subsystems.
5. **Deterministic permission outcomes**
   - Same request + scope + policy should produce same decision.

---

## UX Principles

1. **One app, many capabilities**
   - Users should not feel backend boundaries.
2. **Permission clarity over hidden magic**
   - Explain what is requested and why.
3. **Fast path for trusted workflows**
   - Allow remembered grants with visibility and revocation.
4. **Safe defaults**
   - Conservative capability profile on first run.
5. **Actionable diagnostics**
   - If something fails, show where and what to do next.

---

## Initial Success Metrics

### Product metrics

- Time-to-first-agentic-task completion
- Successful task execution rate for local filesystem tasks
- Permission prompt accept/deny rates and churn
- User retention after first desktop onboarding

### Engineering metrics

- Crash-free sessions
- Sidecar startup success rate
- Mean time to recover from sidecar failure
- Tool-policy consistency pass rate in integration tests

---

## Open Questions

1. Which process owns canonical auth/session identity between OpenWebUI and OpenCode?
2. Should OpenHubUI broker all privileged operations, or only MCP/tool-related operations?
3. Do we support remote OpenWebUI targets in MVP, or only local/managed targets?
4. What is the minimum viable permission UX that remains safe but not noisy?
5. What compatibility SLA do we set for OpenCode version upgrades?

---

## Recommended Next Actions

1. Create a companion `electron-process-model.md` with concrete process/port map.
2. Create a `security-threat-model.md` focused on IPC, sandboxing, and secret handling.
3. Create an `integration-contracts.md` defining OpenWebUI ↔ OpenCode ↔ OpenHubUI request/response schemas.
4. Define MVP acceptance criteria and test matrix before implementation begins.

---

## Summary

OpenHubUI as a desktop application built with Electron and integrated with OpenCode is a strong strategic direction. The highest-leverage approach is to:

- Keep OpenHubUI’s identity centered on MCP aggregation and control plane,
- Use OpenCode as compatibility/runtime leverage,
- Preserve OpenWebUI as the familiar user experience,
- Enforce security through a strict brokered capability model.

This composition can deliver a practical local-agent workspace that addresses server rigidity while enabling advanced, safe, and extensible agentic workflows.
