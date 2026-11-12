#!/usr/bin/env bash
# Print whether a release version should be marked as a GitHub pre-release.
#
# Every 0.x release is a pre-release: the public API can still change, and the
# repository's "Latest" release must not advertise one as stable. A SemVer
# pre-release suffix is also unstable, including before a stable release.
set -euo pipefail

version=${1:?usage: release-prerelease.sh VERSION}
version=${version#v}

case "$version" in
  0.* | *-*) echo true ;;
  *) echo false ;;
esac
