"""Run on both machines and compare the JSON: python scripts/diagnose_setup.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pronunciationcoach.diagnostics import runtime_snapshot

if __name__ == "__main__":
    print(json.dumps(runtime_snapshot(), indent=2))
