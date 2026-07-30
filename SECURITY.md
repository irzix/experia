# Security Policy

Experia treats the confidentiality of vulnerability reports as a first-class
concern. This policy applies project-wide to every module, integration,
example, and released artifact in this repository.

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Report privately
through GitHub's private vulnerability reporting for this repository, which is
visible only to the maintainers:

- **Private reporting channel:** https://github.com/irzix/experia/security/advisories/new

If you cannot access GitHub private advisories, open a regular
[issue](https://github.com/irzix/experia/issues) that only asks the maintainers
to contact you privately, without disclosing any vulnerability details.

## What to include

- The affected Experia version and release line.
- The affected component or integration.
- A minimal reproduction and the observed impact.
- Any known mitigation or workaround.

## Response commitments

- **Acknowledgement target:** at most 3 UTC business days from receipt.
- **Scope:** project-wide, covering all packages, integrations, and published artifacts.

After acknowledgement, the maintainers triage the report privately, agree on a
remediation and disclosure timeline with the reporter, and coordinate a fix and
advisory before any public disclosure. Reporter identity is kept private unless
the reporter asks to be credited.

## Supported release lines

Security fixes are provided only for supported release lines. Older lines are
end-of-life and should upgrade to the current line.

- **Supported release lines:** 0.8.x: supported; 0.7.x: supported; 0.2.x: end-of-life; 0.1.x: end-of-life

| Release line | Status | Security fixes |
| ------------ | ------------ | -------------- |
| 0.8.x | Supported | Yes |
| 0.7.x | Supported | Yes |
| 0.2.x | End-of-life | No |
| 0.1.x | End-of-life | No |

The current release line tracks the version declared in `pyproject.toml`. When a
new minor line is released, this table and the machine-readable
`Supported release lines` field are updated in the same change set.
