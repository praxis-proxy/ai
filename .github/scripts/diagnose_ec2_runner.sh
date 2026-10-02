#!/usr/bin/env bash
# Report safe diagnostics while a failed GPU runner is still running.
set -euo pipefail

diag_dir="$(mktemp -d "${RUNNER_TEMP:-/tmp}/ec2-runner-diagnostics.XXXXXX")"
trap 'rm -rf "$diag_dir"' EXIT

if ! aws ec2 describe-instance-status \
  --instance-ids "$INSTANCE_ID" --include-all-instances \
  --query 'InstanceStatuses[].{State:InstanceState.Name,SystemCheck:SystemStatus.Status,InstanceCheck:InstanceStatus.Status}' \
  --output json; then
  echo '::warning::Could not read EC2 instance status checks'
fi

# User data contains a short-lived registration token. Keep all raw boot output
# in a temporary file and print only fixed markers, never its original lines.
console_file="$diag_dir/console.txt"
if aws ec2 get-console-output --instance-id "$INSTANCE_ID" --latest \
  --query Output --output text > "$console_file" 2> "$diag_dir/console-error.txt"; then
  python3 - "$console_file" <<'PY'
from pathlib import Path
import re
import sys

output = Path(sys.argv[1]).read_text(errors="replace")
if not output.strip() or output.strip() == "None":
    print("EC2 console output is not available yet")
else:
    print(f"EC2 console output available ({len(output)} characters; raw output withheld)")
    markers = {
        "cloud-init completed": r"Cloud-init[^\n]*finished",
        "network or download error": r"curl: \(\d+\)|Could not resolve host|Connection timed out|SSL connection error",
        "runner version rejected": r"runner version is out of date",
        "runner connected": r"Connected to GitHub",
    }
    for name, pattern in markers.items():
        print(f"{name}: {'yes' if re.search(pattern, output, re.IGNORECASE) else 'no'}")
PY
elif grep -Eq 'AccessDenied|UnauthorizedOperation' "$diag_dir/console-error.txt"; then
  echo '::warning::EC2 console output denied; grant ec2:GetConsoleOutput to the runner start role'
else
  echo '::warning::Could not retrieve EC2 console output'
fi

# The pinned action suppresses errors from this API while polling. Query it
# again with the same PAT so permission failures are visible separately from
# a runner that never registered.
if gh api "repos/$GITHUB_REPOSITORY/actions/runners?per_page=100" \
  > "$diag_dir/runners.json" 2> "$diag_dir/runners-error.txt"; then
  runner_status="$(jq -r --arg label "$RUNNER_LABEL" \
    '.runners[]? | select(any(.labels[]?; .name == $label)) | .status' \
    "$diag_dir/runners.json")"
  if [[ -n "$runner_status" ]]; then
    echo "Runner label is present in GitHub with status: $runner_status"
  else
    echo 'Runner label is absent from the first page of GitHub runners'
  fi
  if (( $(jq -r '.total_count' "$diag_dir/runners.json") > 100 )); then
    echo '::warning::Runner list has more than 100 entries; first-page result may be incomplete'
  fi
else
  http_error="$(grep -oE 'HTTP [0-9]{3}' "$diag_dir/runners-error.txt" | head -1 || true)"
  echo "::warning::Runner PAT could not list GitHub runners${http_error:+ ($http_error)}"
fi
