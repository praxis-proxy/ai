#!/usr/bin/env bash
# Print whether a release version should be marked as a GitHub pre-release.
#
# Every 0.x release is a pre-release: the public API can still change, and the
# repository's "Latest" release must not advertise one as stable.
set -euo pipefail

version=${1:?usage: release-prerelease.sh VERSION}
version=${version#v}

if [ "${version%%.*}" = "0" ]; then
  echo true
else
  echo false
fi
