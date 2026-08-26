# Release Process

## Versioning

Praxis AI uses [Semantic Versioning][semver]. The
workspace version is the single source of truth, defined
in `workspace.package.version` in the root `Cargo.toml`.
All workspace crates inherit this version.

[semver]: https://semver.org/

## Distribution

The supported release artifact is the Praxis AI container image on GHCR.
Workspace crates are implementation packages and are not published to
crates.io independently.

## Pre-release Checklist

Before tagging a release:

- [ ] Lints are clean (`make lint`)
- [ ] All tests pass locally (`make test`)
- [ ] Dependency audit passes (`make audit`)
- [ ] Benchmarks have been run; performance is similar
      or better than the previous release
- [ ] Version in root `Cargo.toml` is bumped
- [ ] `Cargo.lock` is regenerated with the new version
- [ ] `SECURITY.md` lists the new minor version
- [ ] Pull request labels produce useful generated release notes

## Tagging a Release

Tags follow the format `v<MAJOR>.<MINOR>.<PATCH>` (e.g.
`v0.1.0`). Push the tag to the repository:

```console
git tag v0.1.0
git push origin v0.1.0
```

## Publishing Container Images

Container images are published to
[GitHub Container Registry][ghcr] (GHCR).

Pushing a valid release tag triggers the **Release** workflow. It verifies
that the tag matches `workspace.package.version`, runs the test suite,
builds the multi-stage Alpine image, pushes it to
`ghcr.io/praxis-proxy/ai`, and creates the GitHub Release.

Reviewers can manually dispatch the **Publish** workflow when a container
image is needed without creating a tagged GitHub Release.

[ghcr]: https://ghcr.io/praxis-proxy/ai

### Image Tags

The release workflow produces these tags per run:

| Pattern | Example | Description |
| --------- | --------- | ------------- |
| `sha-<hash>` | `sha-abc1234` | Git commit SHA |
| `<version>` | `0.1.0` | Full semver (from git tag) |
| `<major>.<minor>` | `0.1` | Major.minor shorthand |

The workflow also publishes a `sha-<hash>` tag for traceability.

## Changelog

Praxis AI uses [GitHub Releases][gh-releases] for changelogs. The release
workflow creates each release with generated notes. Review pull request
labels before tagging so entries fall into the categories configured in
`.github/release.yml`. There is no separate `CHANGELOG.md` file.

[gh-releases]: https://github.com/praxis-proxy/ai/releases

## Release Branches

Release branches are optional and created from tags when
backports are needed. The naming convention is
`release/v<MAJOR>.<MINOR>.x` (e.g. `release/v0.1.x`).

Fixes are cherry-picked onto the release branch and a new patch tag is
created from it. The tag triggers the release workflow as usual.

## Container Details

The production image is a minimal Alpine container:

- Static musl build with LTO, single codegen unit,
  and stripped symbols
- Runs as non-root user (`praxis`)
- Exposes ports `8080` (proxy) and `9901` (admin)
- Built-in health check at
  `http://127.0.0.1:9901/healthy`
- Config directory and working directory: `/etc/praxis`

> **Note**: This is subject to change.
