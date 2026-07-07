<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Praxis Contributors
-->

# Stored-session replay fixtures

These fixtures are sanitized examples shaped like stored agent sessions.
Each fixture includes one or more ordered turns with the request sent by
an agent client and the response returned by the mocked upstream model
service.

The samples are intentionally small. They exercise the Praxis example
configuration paths for Anthropic Messages and OpenAI Responses while
leaving room for future import tooling that can normalize real Claude
or Codex session logs into the same fixture schema.
