#!/usr/bin/env python3
"""Git custom merge driver for the pipeline's self-tracked usage-counter
files (tavily_usage.json, google_tts_usage*.json, gemini_image_usage*.json).

These are single-line {"month"/"day": ..., "calls"/"characters"/"requests": N}
files that get incremented by every pipeline run. A plain three-way text merge
can't combine two independent increments of the same line -- it always
conflicts, even though the correct result is obviously "add both deltas to the
common base". Registered via .gitattributes + `git config merge.usage-counter.driver`.

Usage (as invoked by git): merge_usage_counter.py %O %A %B
  %O = common ancestor version, %A = current branch's version (also the
  output path git expects the merge result written to), %B = other branch's
  version.
"""
import json
import sys

PERIOD_KEYS = ("month", "day")
COUNT_KEYS = ("calls", "characters", "requests")


def _load(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main() -> int:
    base_path, ours_path, theirs_path = sys.argv[1], sys.argv[2], sys.argv[3]
    base, ours, theirs = _load(base_path), _load(ours_path), _load(theirs_path)

    period_key = next(k for k in PERIOD_KEYS if k in ours or k in theirs)
    count_key = next(k for k in COUNT_KEYS if k in ours or k in theirs)

    base_val = base.get(count_key, 0)
    ours_val = ours.get(count_key, base_val)
    theirs_val = theirs.get(count_key, base_val)

    ours_period = ours.get(period_key)
    theirs_period = theirs.get(period_key)
    if ours_period != theirs_period:
        # A month/day boundary rolled over on one side since the common
        # ancestor -- take whichever side is in the later period as-is rather
        # than summing counts from two different billing periods.
        period, merged_val = max((ours_period, ours_val), (theirs_period, theirs_val))
    else:
        period = ours_period
        merged_val = base_val + (ours_val - base_val) + (theirs_val - base_val)

    with open(ours_path, "w") as f:
        json.dump({period_key: period, count_key: merged_val}, f)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
