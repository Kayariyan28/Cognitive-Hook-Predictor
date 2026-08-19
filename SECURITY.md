# Security policy

## Supported version

Security fixes are applied to the latest revision of the default branch. This
research-oriented project does not currently maintain older release branches.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability, leaked credential, or
exposed creator video. Use the repository's **Security → Report a
vulnerability** flow to submit a private GitHub security advisory. Include:

- the affected revision and component;
- a minimal reproduction or request trace with secrets removed;
- the likely impact and prerequisites; and
- any suggested mitigation.

If private vulnerability reporting is not enabled, contact the repository owner
through their GitHub profile without publishing exploit details. Please allow
the maintainer time to reproduce and address the report before disclosure.

## Deployment boundary

The development launchers bind the services to loopback and do not implement
user authentication. Do not expose the development server directly to the
internet. A production deployment needs, at minimum, TLS, authentication,
authorization, request-size and rate limits, isolated worker execution,
restricted CORS, logging with sensitive-data redaction, and a documented data
retention/deletion policy.

Uploaded clips, extracted thumbnails, job records, model caches, and cortical
results can be sensitive. Local runtime directories are ignored by Git, but
operators remain responsible for filesystem permissions, backups, retention,
and secure deletion.

Never commit `.env.local`, Hugging Face tokens, worker bearer tokens, model
registry secrets, creator media, or inference results. Revoke and rotate any
credential immediately if it is exposed.

## Model-safety boundary

TRIBE v2 output is a predicted average-subject cortical BOLD tensor, not a
diagnosis or a measured response from a person. SignalFrame is not a medical
device. Security reports should distinguish code vulnerabilities from model
accuracy, calibration, licensing, or scientific-validity concerns.
