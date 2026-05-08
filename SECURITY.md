# Security Policy

## Supported Versions

We release security updates for the latest minor version series. Please keep your LogosDB installation up to date.

| Version | Supported          |
|---------| ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in LogosDB, please report it responsibly:

**Email**: security@logosdb.dev (or the repository owner's contact if this is a fork)

Please include:
- Description of the vulnerability
- Steps to reproduce (if applicable)
- Potential impact assessment
- Suggested fix (if you have one)

We aim to:
- Acknowledge receipt within 48 hours
- Provide a timeline for a fix within 7 days
- Coordinate disclosure once a patch is available

## What to Report

Security issues include but are not limited to:

- Memory safety issues (buffer overflows, use-after-free)
- Injection vulnerabilities in CLI or MCP server
- Path traversal or file system issues
- Denial of service via malformed inputs
- Information disclosure via logs or error messages

## What NOT to Report

- General bugs (use [Issues](https://github.com/jose-compu/logosdb/issues) instead)
- Feature requests (use [Issues](https://github.com/jose-compu/logosdb/issues) instead)
- Performance issues (unless they constitute DoS)

## Best Practices for Users

- Keep LogosDB updated to the latest version
- Run with minimal privileges
- Validate inputs when embedding LogosDB in larger systems
- Review MCP server configuration for your threat model

## Disclosure Policy

We follow a coordinated disclosure approach:

1. Report received and acknowledged
2. Fix developed and tested
3. Patch released
4. Public disclosure after users have time to update

We appreciate responsible disclosure and will credit reporters (with permission) in release notes.
