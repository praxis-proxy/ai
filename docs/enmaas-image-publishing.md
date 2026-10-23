# EnMaaS image publishing

This fork publishes the EnMaaS deployment image separately from the upstream
project's release images. The upstream GHCR release process remains documented
in [the release process](release.md); this document covers only the fork-side
Quay image.

The image is built by `.github/workflows/publish-quay.yaml` on every push to
`main` and pushed to:

```text
quay.io/modden/praxis-ai
```

This is an **interim personal namespace** owned by `@mrodden`. It remains in
place until the planned build and deployment migration. Moving to an
org-owned Quay repository later requires changing the workflow `IMAGE` value,
updating the GitHub Actions secrets, and updating the cluster pull secret.

## Build configuration

The workflow must keep:

```text
PRAXIS_AI_FEATURES=full,gcp-adc-filter
```

This matches the production deployment build. The `gcp-adc-filter` feature is
required for the production GCP ADC credential filter; production routing
breaks without it. The build uses GitHub Actions cache to avoid rebuilding the
Rust dependency graph on every push.

## Ownership

- Quay repository: `quay.io/modden/praxis-ai`
- Owner: Mathew Odden (`@mrodden`)
- Intended long-term target: an org-owned Quay repository or the planned
  build pipeline

The personal namespace is an interim build path, not a production
supply-chain boundary. Production deployments must pin the digest from an
approved workflow run.

## Credentials

Two separate robot accounts are used:

- **Build/push:** one repository-scoped robot per image repository, with
  `Write` permission and no broader access. This repository's builder
  credentials are stored only as the `QUAY_USERNAME` and `QUAY_TOKEN` GitHub
  Actions repository secrets and are used only by this workflow.
- **Pull:** one separate read-only robot with `Read` permission on the related
  EnMaaS image repositories. Its credentials are stored only in the cluster
  pull secrets. It is not used by CI.

The workflow uses `docker/login-action`; the builder credentials are passed as
action inputs and are never interpolated into a shell command.

## Rotation

Rotate a credential immediately if it is exposed or a maintainer with access
leaves, and otherwise on a regular schedule (at least every 90 days).

1. Create a replacement robot token in Quay with the same scope.
2. Update the single location that holds the old credential:
   - builder: the `QUAY_USERNAME` and `QUAY_TOKEN` repository secrets;
   - pull: the cluster pull secrets.
3. Run the publish workflow and verify the image pushes, or verify a test pod
   can pull the image.
4. Revoke the old robot token.

## Release and consumption policy

- Every push to `main` produces a development image:
  - `:main` is a mutable convenience tag.
  - `:sha-<full 40-character commit SHA>` identifies the build for that commit.
- Tags are convenience pointers. The image digest is the immutable reference.
- The workflow records `quay.io/modden/praxis-ai@sha256:...` plus the source
  commit in the Actions run summary. That digest is the authoritative
  deployment reference.
- Deploy by digest. Never deploy `:main`.
- Promotion is explicit: build on `main`, validate the digest in staging, then
  update the digest pin in the EnMaaS deployment configuration for staging and
  production.
- Publishing is independent of the upstream Tests workflow, which runs
  separately on `main`. The publish job does not wait for it.
- Version tags and signed provenance attestations are deferred until the
  planned build migration. The OCI source, revision, and version labels
  provide traceability metadata only; they are not a signature.
