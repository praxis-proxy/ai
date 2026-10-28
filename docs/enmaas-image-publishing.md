# EnMaaS image publishing

This fork publishes the EnMaaS deployment image separately from the upstream
project's release images. The upstream GHCR release process remains documented
in [the release process](release.md); this document covers only the fork-side
Quay image.

The image is built by `.github/workflows/publish-quay.yaml` on every push to
`main`, with a manual dispatch path for rebuilding a specific `main` commit,
and pushed to:

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

## Triggers and manual rebuilds

- A push to `main` builds the image and updates both `:main` and
  `:sha-<commit>` tags.
- `workflow_dispatch` accepts an optional `commit` input. A manual rebuild
  tags only `:sha-<commit>`; it never moves `:main`.
- The workflow is intended for branches in the canonical repository, where
  the Quay builder secrets are available. Do not expose those credentials to
  arbitrary fork code.

## SBOM artifacts

Every successful build generates two SPDX JSON documents for the source
commit:

- `image.spdx.json`: the packages visible in the exact pushed image digest.
- `source.spdx.json`: the source tree and Rust dependency graph from
  `Cargo.lock`; this is where Rust crates are enumerated even when the
  runtime binary does not carry embedded cargo metadata.

Both files are uploaded together as the Actions artifact
`praxis-ai-sbom-<commit>` with 90-day retention. The job summary records the
image digest, source commit, and artifact name.

Download the artifact from the workflow run, for example:

```bash
gh run download <run-id> -n praxis-ai-sbom-<commit>
```

The SBOM is inventory evidence, not a signature. Signed provenance and
attestations are still deferred; deploy by digest and retain the SBOM
artifact with the deployment record.

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
  - `:main` is a mutable convenience tag, updated only by a `main` push.
  - `:sha-<full 40-character commit SHA>` identifies the build for that commit.
- Manual dispatches produce only the `:sha-` tag for the selected commit.
- Tags are convenience pointers. The image digest is the immutable reference.
- The workflow records `quay.io/modden/praxis-ai@sha256:...`, the source
  commit, and the SBOM artifact name in the Actions run summary. That digest
  is the authoritative deployment reference.
- Deploy by digest. Never deploy `:main`.
- Promotion is explicit: build on `main`, validate the digest in staging, then
  update the digest pin in the EnMaaS deployment configuration for staging and
  production.
- Publishing is independent of the upstream Tests workflow, which runs
  separately on `main`. The publish job does not wait for it.
- Version tags and signed provenance attestations are deferred until the
  planned build migration. The OCI source, revision, and version labels
  provide traceability metadata only; they are not a signature.
