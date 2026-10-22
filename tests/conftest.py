"""Shared pytest fixtures.

Design goal: the whole suite runs in a few seconds with zero network calls and
zero real ffmpeg/TTS invocations, so it's cheap enough to run before every
commit (see scripts/run_tests.sh). Tests that genuinely need real ffmpeg are
marked `integration` and excluded by default (see pytest.ini).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config, fundamentals  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Redirects CACHE_DIR/OUTPUT_DIR to a per-test tmp dir so tests never read
    or write the real cache/output folders (avoids cross-test pollution and
    accidental interference with real generated videos)."""
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    cache_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(config, "TAVILY_USAGE_FILE", tmp_path / "tavily_usage.json")
    monkeypatch.setattr(config, "VISUAL_USAGE_FILE", tmp_path / "visual_usage.json")
    monkeypatch.setattr(config, "GEMINI_IMAGE_USAGE_FILE", tmp_path / "gemini_image_usage.json")
    monkeypatch.setattr(config, "CLOUDFLARE_IMAGE_USAGE_FILE", tmp_path / "cloudflare_image_usage.json")
    yield cache_dir, output_dir


@pytest.fixture(autouse=True)
def _company_symbol_search_no_match_by_default(monkeypatch):
    """research.resolve_symbol_for_theme / resolve_company_for_theme (used by
    both pipeline/research.py and pipeline/prompt_script.py) can call
    fundamentals.search_company_symbol or fundamentals.search_company, real
    network requests against Screener.in's search endpoint, for any
    finance-ish theme string that doesn't name an explicit "NSE: SYMBOL".
    Default both to "no match" so tests that don't care about this resolution
    path never silently hit the real network. Tests exercising the resolution
    path override this explicitly."""
    monkeypatch.setattr(fundamentals, "search_company_symbol", lambda name: "")
    monkeypatch.setattr(fundamentals, "search_company", lambda name: {})


@pytest.fixture(autouse=True)
def _google_news_fallback_empty_by_default(monkeypatch):
    """research.search_web's last-resort Google News RSS fetch must never hit
    the real network from tests; tests of it call the real implementation
    captured at import time in tests/test_research.py."""
    from pipeline import research
    monkeypatch.setattr(research, "_fetch_google_news_facts", lambda topic: "")


@pytest.fixture(autouse=True)
def _no_company_insights_network_by_default(monkeypatch):
    """build_theme_entry and prompt_script's freeform path look up a sourced
    story fact (Wikipedia/web search) and Screener's peer table via
    company_insights.did_you_know_facts; default every entry point to "nothing
    found" so tests never hit the network. Tests exercising the did-you-know
    behaviour re-point these at the real functions (captured at import time in
    their own module) or at their own fakes."""
    from pipeline import company_insights, prompt_script
    monkeypatch.setattr(company_insights, "did_you_know_facts", lambda name, data: [])
    monkeypatch.setattr(prompt_script, "_find_did_you_know", lambda resolved, symbol: {"facts": [], "narration": ""})
    monkeypatch.setattr(company_insights, "articulate_facts", lambda name, facts: "")
    # The faithfulness second opinion is its own LLM call; tests mocking a
    # single chat_json reply would otherwise feed it the wrong payload.
    monkeypatch.setattr(company_insights, "_is_faithful", lambda claim, source: True)
    monkeypatch.setattr(fundamentals, "fetch_peers", lambda data: [])


@pytest.fixture
def no_api_keys(monkeypatch):
    """Explicitly blanks API keys for tests asserting the "not configured" error paths."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "PEXELS_API_KEY", "")
