#!/usr/bin/env python3
"""Git custom merge driver for the pipeline's self-tracked usage-counter
files (tavily_usage.json, google_tts_usage*.json).

These are single-line {"month": ..., "calls"/"characters": N} files that get
incremented by every pipeline run. A plain three-way text merge can't combine
two independent increments of the same line -- it always conflicts, even
though the correct result is obviously "add both deltas to the common base".
Registered via .gitattributes + `git config merge.usage-counter.driver`.

Usage (as invoked by git): merge_usage_counter.py %O %A %B
  %O = common ancestor version, %A = current branch's version (also the
  output path git expects the merge result written to), %B = other branch's
  version.
"""
import json
import sys


def _load(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main() -> int:
    base_path, ours_path, theirs_path = sys.argv[1], sys.argv[2], sys.argv[3]
    base, ours, theirs = _load(base_path), _load(ours_path), _load(theirs_path)

    key = "calls" if "calls" in ours or "calls" in theirs else "characters"
    base_val = base.get(key, 0)
    ours_val = ours.get(key, base_val)
    theirs_val = theirs.get(key, base_val)

    ours_month = ours.get("month")
    theirs_month = theirs.get("month")
    if ours_month != theirs_month:
        # A month boundary rolled over on one side since the common ancestor --
        # take whichever side is in the later month as-is rather than summing
        # counts from two different billing periods.
        month, merged_val = max((ours_month, ours_val), (theirs_month, theirs_val))
    else:
        month = ours_month
        merged_val = base_val + (ours_val - base_val) + (theirs_val - base_val)

    with open(ours_path, "w") as f:
        json.dump({"month": month, key: merged_val}, f)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
