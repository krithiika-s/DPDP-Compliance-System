"""Print the non-out_of_scope findings of saved run files (status, category, reason code, claim start).

Usage (repo root):
    python scripts\\compare_runs.py before_violating_1.json after_violating_1.json
Only claim labels cut to 60 characters are printed; no policy text.
"""
import json
import sys
from pathlib import Path


def rows(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in data["alignment"]["findings"]:
        if f["status"] == "out_of_scope":
            continue
        claim = (f.get("claim") or {}).get("claim") or "(undeclared behaviour)"
        out.append((f["category"], f["status"], f.get("reason_code", "-"), claim[:60]))
    return sorted(out)


def main() -> None:
    for path in sys.argv[1:]:
        info = json.loads(Path(path).read_text(encoding="utf-8")).get("run_info", {})
        print(f"\n=== {path}  (commit {info.get('commit', 'n/a')})")
        for cat, status, reason, claim in rows(path):
            print(f"  {status:12} {cat:22} {reason:22} {claim}")


if __name__ == "__main__":
    main()