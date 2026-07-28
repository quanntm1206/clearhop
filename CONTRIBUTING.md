# Contributing

## Development

Use Python 3.10 or newer. Create an isolated environment, then install:

```powershell
python -m pip install -e ".[desktop,dev,metrics,export]"
python -m pytest -q
python scripts/verify.py --publish-readiness
```

Changes to inference require focused tests, the full suite, production and research verifiers, and a generated-audio smoke. Never commit private audio, datasets, checkpoints, credentials, or unreviewed third-party weights.

## Pull requests

Describe the contract changed, validation commands, and evidence paths. Keep external claims separate from reproduced local results. Report unavailable dependencies as `blocked`, never as pass.
