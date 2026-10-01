"""Real-ffmpeg end-to-end checks. Excluded from the default `pytest` run (see
pytest.ini) since they shell out to actual ffmpeg encodes — run explicitly
with `pytest -m integration` (e.g. before a release, or when touching
pipeline/assemble.py) rather than on every commit.
"""
import shutil

import pytest

from pipeline import assemble

pytestmark = pytest.mark.integration

FFMPEG_AVAILABLE = shutil.which("ffmpeg") and shutil.which("ffprobe")


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not installed")
def test_assemble_video_real_ffmpeg_audio_starts_at_t0(tmp_path):
    """Regression test for the 'start narration at t=0' fix: the final mux must
    play audio under the hook card, not wait for it to finish."""
    import subprocess

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # 1x1 synthetic color image as the hook card + scene visual.
    hook_card = tmp_path / "hook.png"
    subprocess.run(
        [assemble.FFMPEG_BIN, "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64", "-frames:v", "1", str(hook_card)],
        check=True, capture_output=True,
    )
    visual = tmp_path / "scene.jpg"
    subprocess.run(
        [assemble.FFMPEG_BIN, "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1", str(visual)],
        check=True, capture_output=True,
    )
    # 2-second silent-but-present audio tone as narration.
    narration = tmp_path / "narration.mp3"
    subprocess.run(
        [assemble.FFMPEG_BIN, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(narration)],
        check=True, capture_output=True,
    )

    out_path = tmp_path / "final.mp4"
    assemble.assemble_video(
        scenes=[{"narration": "test", "on_screen_text": ""}],
        narration_paths=[narration],
        visual_paths=[(visual, "image")],
        out_path=out_path,
        work_dir=work_dir,
        width=64, height=64,
        hook_card_path=hook_card,
    )

    assert out_path.exists()
    probe = subprocess.run(
        [assemble.FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)],
        check=True, capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())
    # hook (2s) + scene (2s, sized to narration duration) = ~4s total.
    assert duration >= 3.5

    volume = subprocess.run(
        [assemble.FFMPEG_BIN, "-i", str(out_path), "-t", "1", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    assert "mean_volume: -inf" not in volume.stderr, "audio must be present in the first second (t=0), not silent"
