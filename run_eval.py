"""
Runs eval/test_cases.json against a running instance of the API (POST /ask)
and writes pass/fail + the actual answer back into eval/results.json.

Pass criteria (simple, source-file based -- not exact-answer matching):
  - If expected_source_files is non-empty: pass if at least one expected
    file appears among the citations returned.
  - If expected_source_files is empty (an out-of-corpus check): pass if
    the API returned zero citations (i.e. it correctly said "not found").

Usage:
    python run_eval.py                       # assumes http://localhost:8000
    python run_eval.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
from pathlib import Path

import requests

TEST_CASES_PATH = Path(__file__).parent / "eval" / "test_cases.json"
RESULTS_PATH = Path(__file__).parent / "eval" / "results.json"


def run(base_url: str) -> None:
    test_cases = json.loads(TEST_CASES_PATH.read_text())

    results = []
    passed = 0

    for case in test_cases:
        question = case["question"]
        expected = set(case.get("expected_source_files", []))

        try:
            resp = requests.post(f"{base_url}/ask", json={"question": question}, timeout=60)
            resp.raise_for_status()
            body = resp.json()
        except Exception as e:
            results.append({**case, "pass": False, "error": str(e)})
            print(f"[FAIL] {case['id']}: request error -- {e}")
            continue

        actual_files = {c["source_file"] for c in body.get("citations", [])}

        if expected:
            ok = bool(expected & actual_files)
        else:
            ok = len(actual_files) == 0

        if ok:
            passed += 1

        results.append(
            {
                **case,
                "pass": ok,
                "actual_answer": body.get("answer"),
                "actual_source_files": sorted(actual_files),
            }
        )
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']}: {question}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n{passed}/{len(test_cases)} passed. Full results written to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    if not TEST_CASES_PATH.exists():
        print(f"No test cases found at {TEST_CASES_PATH}", file=sys.stderr)
        sys.exit(1)

    run(args.base_url)
