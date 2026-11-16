#!/usr/bin/env python3
"""Require completed release gates before publication with repaired tooling."""

import argparse
import re
import subprocess
import tomllib
from pathlib import Path

from release_qualification import api, eligible, pages


GATES = {"validate", "test", "container", "container-experimental", "container-fips", "fips-proof"}


def verify_run(repo, tag, sha, run, jobs):
    """Allow only the original tag's successful gates in the latest attempt."""
    if not re.fullmatch(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", tag):
        raise ValueError("recovery requires a plain vMAJOR.MINOR.PATCH tag")
    if (not isinstance(run, dict) or not eligible(run, repo, sha, "release.yaml")
            or run.get("head_branch") != tag or run.get("status") != "completed"):
        raise ValueError("release run must be completed and identify the repository, tag, and commit")
    if run.get("event") not in ("push", "workflow_dispatch"):
        raise ValueError("release run must originate from a tag push or dispatch")
    if any(job.get("run_id") != run["id"] or job.get("head_sha") != sha
           or job.get("run_attempt") != run["run_attempt"] for job in jobs):
        raise ValueError("release jobs must belong to the selected run, commit, and latest attempt")
    for name in GATES:
        matches = [job for job in jobs if job.get("name") == name]
        if (len(matches) != 1 or matches[0].get("status") != "completed"
                or matches[0].get("conclusion") != "success"):
            raise ValueError(f"release gate {name} must have completed successfully")
    # A publication failure is recoverable; failures elsewhere are not.
    if any(job.get("name") != "github-release" and job.get("conclusion") != "success" for job in jobs):
        raise ValueError("only the github-release job may have failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD^{commit}"], text=True).strip()
    version = tomllib.loads(Path("Cargo.toml").read_text())["workspace"]["package"]["version"]
    if args.tag != "v" + version:
        parser.error("tag must match the checked-out workspace version")
    try:
        run = api(f"repos/{args.repo}/actions/runs/{args.run_id}")
        if not isinstance(run, dict) or run.get("id") != args.run_id:
            raise ValueError("release run lookup failed")
        jobs = pages(f"repos/{args.repo}/actions/runs/{args.run_id}/attempts/{run['run_attempt']}/jobs", "jobs")
        verify_run(args.repo, args.tag, sha, run, jobs)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"attempt={run['run_attempt']}")


if __name__ == "__main__":
    main()
