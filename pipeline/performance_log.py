"""Records which real, uploaded YouTube video_id corresponds to which NSE
company, so a future performance-feedback feature (e.g. biasing trending-stock
selection toward market-cap tiers that have historically performed better --
see pipeline/trending_stocks.py) can join this channel's own view/engagement
data back to "which company was this video about" with zero fuzzy-matching
risk. video_id is captured in-process immediately after a successful upload
(youtube_upload.upload_video()'s own return value), so this is an exact,
non-fuzzy key -- never a regex/guess against the LLM-generated title, which
can truncate, paraphrase, or abbreviate the company name.

Best-effort throughout, matching every other usage/analytics file in this
pipeline (tavily_usage.json, google_tts_usage.json): a read/write failure here
must never block or fail video production.
"""
import json

from . import config


def record(video_id: str, symbol: str, company_name: str) -> None:
    """Appends one entry to the performance log. No-op on any failure."""
    if not video_id or not symbol:
        return
    try:
        entries = load()
        entries.append({"video_id": video_id, "symbol": symbol, "company_name": company_name})
        config.COMPANY_PERFORMANCE_LOG_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except Exception:
        pass


def load() -> list:
    """Returns [] if the log doesn't exist yet or is unreadable/corrupt."""
    try:
        return json.loads(config.COMPANY_PERFORMANCE_LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
