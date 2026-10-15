# Project Management

All repositories in the `praxis-proxy` organization
use a consistent workflow for planning, prioritizing,
and tracking work.

## Triage

Every issue goes through triage before it becomes
accepted work. New issues are automatically labeled
`triage/needs-triage` when opened. Reviewers review
incoming issues regularly (typically daily) to assess
scope, validity, and priority.

To accept an issue, assign it to a milestone. Milestone
assignment signals that the issue is understood, scoped,
and planned for work. A GitHub Actions workflow
automatically swaps the label to `triage/accepted` when
a milestone is set. Removing an issue from its milestone
reverts it to `triage/needs-triage`.

| Label | Meaning |
| --- | --- |
| `triage/needs-triage` | Awaiting reviewer review |
| `triage/accepted` | Assigned to a milestone; accepted for work |

## Bot-Generated Issues

A weekly bot opens issues on the repository to surface
potential improvements, bugs, or maintenance tasks. These
issues are labeled `NEED HUMAN REVIEW` and **must not be
worked on until a reviewer has reviewed them**.

### Rules

- **Do not self-assign** a bot-generated issue before it
  has been reviewed.
- **Do not submit a PR** for a bot-generated issue before
  it has been validated by a reviewer.

### Review Process

A reviewer reviews the issue and applies one of the
following labels:

| Label | Meaning |
| --- | --- |
| `NEED HUMAN REVIEW` | Awaiting reviewer review (applied automatically by the bot) |
| `HUMAN REVIEWED` | A reviewer has validated the issue |

To **approve** an issue, replace the `NEED HUMAN REVIEW`
label with `HUMAN REVIEWED`. The label history records
who approved and when.

To **reject** an issue, remove the `NEED HUMAN REVIEW`
label, leave a comment explaining why, and close the
issue.

Once an issue is approved, it follows the normal triage
workflow (milestone assignment, priority, sizing).

## Milestones

Milestones represent a body of work toward a shared
goal (e.g. a release, a feature area, or a hardening
pass). Every issue and pull request should belong to
a milestone. Milestones provide scope boundaries and
help answer "what ships together?"

## Priority

Every issue should have a priority set via the
built-in Priority issue field (not labels). Address
work in priority order:

| Priority | Description |
| --- | --- |
| Urgent | Must be worked on immediately before anything else |
| High | Needs to be worked on immediately, defer to urgents |
| Medium | Resolve after high and urgent |
| Low | Resolve after all other priority levels |

## Size

Every issue should have a size set via the built-in
Size issue field. Size is a rough effort estimate:

| Size | Rough Estimate |
| --- | --- |
| Large | 1 week or more |
| Medium | Roughly 3 days |
| Small | Roughly 1 day |
| Tiny | Less than a day |

## Project Boards

GitHub project boards visualize the state of work
across milestones. Use boards to track issues through
their lifecycle (backlog, in progress, in review,
done). Boards are the primary tool for stand-ups and
status checks.

## Pull Request Assignees

Every pull request gets two assignees once it is ready
for review: a maintainer from `ai-maintainers` or
`project-leads`, and a reviewer from `ai-reviewers`. A
bot assigns them when the PR is opened, reopened, or
marked ready for review, so drafts wait until they are
marked ready.

The bot never picks the PR author, never gives both
roles to the same person, and goes for whoever is
assigned to the fewest open PRs, so the review load
stays spread out. If someone from the right team is
already assigned, it counts them and only fills the
role that is missing.

Changing the assignees by hand is fine. The bot never
removes anyone, and it only runs again if the PR is
reopened, or goes back to draft and is marked ready.
See [pull-request-assignees.yaml] for the details.

[pull-request-assignees.yaml]: ../../.github/workflows/pull-request-assignees.yaml
