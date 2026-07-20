# Project Management

All repositories in the `praxis-proxy` organization
use a consistent workflow for planning, prioritizing,
and tracking work.

## Triage

Every issue goes through triage before it becomes
accepted work. New issues are automatically labeled
`triage/needs-triage` when opened. Maintainers review
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
| `triage/needs-triage` | Awaiting maintainer review |
| `triage/accepted` | Assigned to a milestone; accepted for work |

## Bot-Generated Issues

A weekly bot opens issues on the repository to surface
potential improvements, bugs, or maintenance tasks. These
issues are labeled `NEED HUMAN REVIEW` and **must not be
worked on until a maintainer has reviewed them**.

### Rules

- **Do not self-assign** a bot-generated issue before it
  has been reviewed.
- **Do not submit a PR** for a bot-generated issue before
  it has been validated by a maintainer.

### Review Process

A maintainer reviews the issue and applies one of the
following labels:

| Label | Meaning |
| --- | --- |
| `NEED HUMAN REVIEW` | Awaiting maintainer review (applied automatically by the bot) |
| `HUMAN REVIEWED` | A maintainer has validated the issue |

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
