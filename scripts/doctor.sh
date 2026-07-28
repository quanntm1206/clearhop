#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
"${PYTHON:-python}" scripts/verify.py --publish-readiness --output reports/generated/publish_readiness.json
