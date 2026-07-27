# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes (Alpha) |

Praxis AI is pre-`1.0.0`. Only the latest patch release of the current
minor line receives security updates. Older pre-release lines become
unsupported when a new minor line is released.

## Reporting a Vulnerability

Please report security vulnerabilities by emailing
`security <at> praxis <dot> fast`. Do not open a public issue.

Include:

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Any potential mitigations you have identified

## Response Timeline

Prior to `v1.0.0`, we coordinate disclosure and remediation timelines
with researchers individually. We will publish a standardized response
timeline before the first stable release.

## Severity Classification

We use the following severity levels:

- **Critical**: Remote code execution, authentication bypass, or data exfiltration without user interaction
- **High**: Denial of service with amplification, privilege escalation, or significant data exposure
- **Medium**: Denial of service requiring sustained effort, information disclosure of limited scope
- **Low**: Issues requiring unlikely configurations or minimal impact

## Safe Harbor

We consider security research conducted in good faith to be authorized. We will not pursue legal action
against researchers who follow this policy and report findings responsibly. In fact, we really appreciate
the help in making Praxis more secure, thank you for your efforts!
