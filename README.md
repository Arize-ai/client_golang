<div align="center">
  <img src="https://storage.googleapis.com/arize-assets/arize-logo-white.jpg" width="600" /><br><br>
</div>

# ⚠️ This repository is archived

Development of the Arize Go SDK has moved. The `client_golang` import path
(`github.com/Arize-ai/client_golang`) is no longer maintained.

| Use case | Go to |
|---|---|
| Start here for an overview of all Go SDKs | **[Arize-ai/client-go](https://github.com/Arize-ai/client-go)** |
| Existing users of this package (ML ingest: predictions, actuals, embeddings, SHAP) | **[Arize-ai/client-go-v1](https://github.com/Arize-ai/client-go-v1)** — drop-in successor, same API surface |
| New projects / typed REST API client (datasets, experiments, prompts, projects, …) | **[Arize-ai/client-go-v2](https://github.com/Arize-ai/client-go-v2)** |

## Migrating from `client_golang`

The v1 successor at [Arize-ai/client-go-v1](https://github.com/Arize-ai/client-go-v1)
preserves the API of this package. Update your import path and run `go mod tidy`:

```diff
- import "github.com/Arize-ai/client_golang"
+ import "github.com/Arize-ai/client-go-v1"
```

```bash
go get github.com/Arize-ai/client-go-v1
go mod tidy
```

No code changes are required for the basic `NewClient` / `Log` flow.

## Why the move?

The package has been split into a documented directory of Go SDKs:

- **client-go** — landing repo with version comparison and migration guidance
- **client-go-v1** — the legacy ML observability ingest client (this package),
  in maintenance mode
- **client-go-v2** — a typed Go client for the Arize REST API, under active
  development

## Status of this repository

This repository is read-only. Issues and pull requests have been disabled.
Existing `go get github.com/Arize-ai/client_golang@<tag>` commands will
continue to resolve because Git tags remain accessible, but no further
releases will be cut here. Please open issues against
[client-go-v1](https://github.com/Arize-ai/client-go-v1/issues) or
[client-go-v2](https://github.com/Arize-ai/client-go-v2/issues) instead.

## Links

- [Arize platform](https://arize.com)
- [Docs](https://docs.arize.com)
- [Slack](https://join.slack.com/t/arize-ai/shared_invite/zt-g9c1j1xs-aQEwOAkU4T2x5K8cqI1Xqg)
