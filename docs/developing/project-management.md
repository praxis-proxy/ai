# Project Management

All repositories in the `praxis-proxy` organization use the same workflow for
planning, prioritizing, and tracking work.

## Triage

Every issue is triaged before it becomes accepted work. New issues are labeled
`triage/needs-triage` automatically. Reviewers check incoming issues
regularly (typically daily) for scope, validity, and priority.

To accept an issue, assign it to a milestone, which says the issue is
understood, scoped, and planned. A GitHub Actions workflow then swaps the
label to `triage/accepted`. Removing the milestone reverts it to
`triage/needs-triage`.

| Label | Meaning |
| --- | --- |
| `triage/needs-triage` | Awaiting reviewer review |
| `triage/accepted` | Assigned to a milestone; accepted for work |

## Bot-Generated Issues

A weekly bot opens issues suggesting improvements, bugs, or maintenance tasks.
They are labeled `NEED HUMAN REVIEW` and **must not be worked on until a
reviewer has reviewed them**.

### Rules

- **Do not self-assign** a bot-generated issue before it has been reviewed.
- **Do not submit a PR** for a bot-generated issue before a reviewer has
  validated it.

### Review Process

| Label | Meaning |
| --- | --- |
| `NEED HUMAN REVIEW` | Awaiting reviewer review (applied automatically by the bot) |
| `HUMAN REVIEWED` | A reviewer has validated the issue |

To **approve** an issue, replace the `NEED HUMAN REVIEW` label with
`HUMAN REVIEWED`; the label history records who approved it and when. To
**reject** one, remove `NEED HUMAN REVIEW`, comment with the reason, and close
the issue. Approved issues then follow the normal triage workflow (milestone,
priority, size).

## Milestones

A milestone is a body of work toward a shared goal, such as a release, a
feature area, or a hardening pass. Every issue and pull request should belong
to one. Milestones define what ships together.

## Priority

Set every issue's priority with the built-in Priority issue field (not
labels), and work in priority order:

| Priority | Description |
| --- | --- |
| Urgent | Must be worked on immediately before anything else |
| High | Needs to be worked on immediately, defer to urgents |
| Medium | Resolve after high and urgent |
| Low | Resolve after all other priority levels |

## Size

Set every issue's size, a rough effort estimate, with the built-in Size issue
field:

| Size | Rough Estimate |
| --- | --- |
| Large | 1 week or more |
| Medium | Roughly 3 days |
| Small | Roughly 1 day |
| Tiny | Less than a day |

## Project Boards

GitHub project boards track issues across milestones through their lifecycle
(backlog, in progress, in review, done). They are the main tool for stand-ups
and status checks.

## Pull Request Review Assignment

Every pull request must link the issue it addresses: `Closes #123` when
merging completes the issue, `Refs #123` when the issue stays open.

Each contributor may have at most three open pull requests ready for review in
this repository; drafts do not count. If opening a ready PR or marking a draft
ready takes the author over the limit, the bot comments on the PR, fails its
limit check, and assigns no reviewer. Once the count is back to three or
fewer, edit or update the waiting PR to rerun the check and reviewer
assignment.

Within the limit, the bot assigns a reviewer from `ai-reviewers` when a PR is
opened, edited, synchronized, reopened, or marked ready for review. It never
picks the PR author, chooses whoever is assigned to the fewest open PRs (ties
broken at random), keeps an already assigned team reviewer, and never removes
assignees. See [pull-request-assignees.yaml] for details.

The bot does not randomly assign maintainers. A maintainer takes ownership
deliberately after all required CI checks pass, and that assignment means
maintainer review has started. Wait for green CI before requesting or
expecting maintainer review.

[pull-request-assignees.yaml]: ../../.github/workflows/pull-request-assignees.yaml
