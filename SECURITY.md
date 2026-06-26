# Security Policy

CNSD is a research framework for machine fault diagnosis. It is not intended for
safety-critical deployment without independent validation (see the limitations
in the README).

## Reporting a vulnerability

If you find a security issue — for example, a way for crafted input data or a
malicious config/YAML file to cause unsafe behavior — please report it privately
rather than opening a public issue.

- Use GitHub's **private vulnerability reporting** (the "Report a vulnerability"
  button under the repository's *Security* tab), or
- Contact a maintainer directly (see MAINTAINERS.md).

Please include a description, steps to reproduce, and the affected version or
commit. We will acknowledge reports and address confirmed issues as promptly as 
we can.

## Supported versions

This is 1.0 research software. Only the latest
commit on `main` is supported; fixes are not backported.

## Scope

CNSD loads user-supplied data and configuration (including YAML). Treat data and
config files from untrusted sources with the same caution as any code — a
configuration file can change which physics and taxonomy the system applies.
