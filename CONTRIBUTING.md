# Contributing to TaxoBranch

Thank you for helping improve TaxoBranch.

## Development setup

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q tests/test_public_api.py tests/test_counterfactual_certificates.py
```

## Pull requests

- Keep the public `child -> parent` edge convention explicit.
- Add a focused test for behavioral changes.
- Do not commit datasets, checkpoints, score tensors, credentials, absolute
  machine paths, or generated scheduler job files.
- Preserve archived protocol semantics. If execution must change without
  changing the scientific contract, document the amendment separately.
- Distinguish feasibility certificates, repair logs, constraint tension,
  optimality gaps, and sensitivity diagnostics.

## Reproducibility reports

Include the exact command, config path, Python and CUDA versions, seed, and the
smallest relevant manifest or traceback. Do not attach licensed datasets or
private credentials.
