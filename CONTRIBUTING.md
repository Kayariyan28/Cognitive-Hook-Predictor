# Contributing

Thank you for helping improve SignalFrame. Keep changes small, testable, and
truthful about what each model can infer.

## Before opening a change

1. Read the setup and verification instructions in [`README.md`](README.md).
2. Search existing issues and describe the user problem before proposing a
   large architectural change.
3. Never add model weights, creator uploads, generated inference artifacts,
   tokens, or machine-specific `.env.local` files.
4. Confirm that every external model, dataset, font, or asset has compatible
   terms and add its provenance to [`NOTICE.md`](NOTICE.md).

## Development checks

For frontend changes, run:

```bash
npm run build
npm test
```

The build runs first because the packaging contract tests inspect generated
files under `dist/`, which is intentionally absent from a clean checkout.

For backend changes, create the Python 3.11 environment described in the README
and run:

```bash
python -m unittest discover -s backend/tests -v
```

Model-dependent smoke tests may require separately downloaded, hash-pinned
artifacts. State clearly when a test could not be run and why.

## Scientific and product invariants

- Never synthesize or decorate cortical activity. The 3D brain may be colored
  only by a verified TRIBE v2 tensor with the exact expected surface contract.
- Keep TRIBE output separate from virality, engagement, retention, attention,
  emotion, memory, and clinical claims.
- Do not expose a behavioral score unless its target, data population,
  evaluation, calibration, and immutable artifact provenance pass the existing
  fail-closed contract.
- Preserve model identifiers, revisions, weight hashes, preprocessing hashes,
  and licensing metadata whenever a model-backed path changes.
- Add or update tests for every contract or behavior change.

## Pull requests

A pull request should explain the problem, the chosen solution, tests run, UI
evidence where relevant, and any data/model/license impact. Keep unrelated
formatting or generated artifacts out of the change. By contributing, you agree
that your contribution is licensed under the repository's MIT License and that
you have the right to submit it.
