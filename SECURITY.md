# GraphMind Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| V5 (Production) | ✅ |
| V4 (Candidate) | ❌ |
| V3 and earlier | ❌ |

## Reporting a Vulnerability

If you discover a security vulnerability in GraphMind, please follow responsible disclosure:

1. **Do NOT open a public GitHub issue.** Security issues should be reported privately.
2. Open a [GitHub Security Advisory](https://github.com/Jdepp007004/GraphMind/security/advisories/new) for this repository.
3. Include as much information as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

We will acknowledge your report within **48 hours** and provide a fix timeline within **7 days**.

## Security Design

GraphMind has a dedicated `SecurityAgent` that enforces:

- **Unknown app handling** — apps not in the known policy set are rejected from the prefetch queue
- **Retention policies** — app data is automatically purged after configurable inactivity periods
- **Event schema validation** — all ingested events are validated against a strict schema before processing
- **Graph boundary enforcement** — the behaviour graph enforces maximum node and edge limits to prevent unbounded memory growth
- **ADB connection sandboxing** — the Android integration layer (ADB connector) operates in read-only telemetry mode

## Data Privacy

GraphMind processes app usage sequences locally on the device. No raw usage data is transmitted off-device. The UbiqLog4UCI dataset used in benchmarking is a publicly available, anonymized research dataset.
