"""Builds an SRT subtitle file from timed narration scenes."""
from pathlib import Path


def _format_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(scenes_with_timing: list[dict], out_path: Path) -> Path:
    """scenes_with_timing: list of {"narration": str, "start": float, "end": float}"""
    lines = []
    for i, scene in enumerate(scenes_with_timing, start=1):
        lines.append(str(i))
        lines.append(f"{_format_ts(scene['start'])} --> {_format_ts(scene['end'])}")
        lines.append(scene["narration"])
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
