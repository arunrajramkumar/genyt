"""Turns a single free-form user prompt into a structured video script via a
hosted LLM (Groq). Generic — works for any topic (stock analysis, history,
science, etc.), not just finance.

If the prompt contains specific facts/figures (e.g. revenue numbers), the model
is instructed to use only those and never invent additional specific claims.
For open-ended topics without hard facts, it can write normally/engagingly.
"""
import json
import re

from . import company_insights, diagrams, fundamentals, llm, prompt_factcheck, research, segmented_script, thumbnail_spec, youtube_insights

SYSTEM_PROMPT = """You write scripts for short-form YouTube videos (including Shorts) on
whatever topic the user gives you. Output ONLY valid JSON (no markdown fences, no
commentary) matching this schema:

{
  "title": "punchy, clickable YouTube title, under 60 characters",
  "description": "2-3 sentence YouTube description",
  "tags": ["8-15 relevant search tags"],
  "scenes": [
    {
      "narration": "1-3 sentences of spoken narration for this scene",
      "visual_query": "2-4 word English search query for stock footage/photos that visually matches this scene",
      "on_screen_text": "SHORT text overlay for this scene, under 6 words, or empty string if none needed",
      "diagram": null
    }
  ]
}

Rules:
- The title must use correct English grammar and punctuation, including
  apostrophes in contractions and possessives (e.g. "What's", "India's").
  Never put hashtags in the title — put them in tags instead.
- For stock/company videos the title is a search query: the company's short name
  first, once (e.g. "Karur Vysya Bank", not "Karur Vysya Bank Ltd"), then the
  concrete event or number from the prompt (e.g. "Karur Vysya Bank Q2 Results:
  Profit +37% | Share Price"). Aim for 60 characters or fewer. Never use a ticker
  symbol, emoji, hashtags, or filler like "Long-Term Wealth", "Wealth Play",
  "Growth Outlook" or "Deep Dive".
- If the prompt contains specific facts, figures, names, or claims, use ONLY
  those — never invent, estimate, or add extra numbers, dates, or claims that were
  not given to you. If the prompt is an analysis of something (e.g. a stock, a
  product, a place), stay neutral and do not add a recommendation/verdict unless
  the user explicitly asked for one. This applies to the title too -- never phrase
  it as a verdict (e.g. avoid "Buy or Sell", "Apply or Avoid", "Cheap or
  Overvalued"); prefer a data-driven, curiosity-led framing instead (e.g.
  "<Subject>: 3 Numbers to Know", "<Subject> at ₹X: What the Numbers Show").
- If the prompt poses an open question (e.g. an exercise/"figure it out" style
  question) without also stating its answer, never invent a resolution — narrate it
  as a question for the viewer to think about instead of asserting a specific answer.
- If the user's prompt is a general/creative topic with no specific facts to
  protect, write naturally and engagingly using your own knowledge.
- Break the narration into scenes so each scene is a single visual beat (aim for
  6-12 scenes for longer videos, 4-6 for short ~30-60s videos).
- Narration must sound natural when read aloud by text-to-speech: no headers, no
  emojis, no markdown.
- Never speak a bare stock ticker/symbol (e.g. "PIDILITIND", "NSE: TCS", "(NSE: CDSL)")
  out loud in narration, since text-to-speech mispronounces it as gibberish. Always
  refer to a company by its full name instead (e.g. say "Pidilite Industries", not
  "PIDILITIND"). A ticker is fine in on_screen_text, just never in narration.
- The first scene (the hook) must lead with the single most dramatic/surprising
  fact or figure from the prompt's content (e.g. a striking percentage, gain, or
  comparison) -- never open with chronological/background setup (e.g. a date, "X
  opened its subscription on...", or the subject's full name) before that figure.
  The hook earns the viewer's attention with the payoff first; background details
  belong later in the narration, not in scene 1.
- Avoid rattling off many distinct numbers back-to-back in one scene or in
  quick succession across consecutive scenes -- when the source content has many
  figures (e.g. issue size, price band, lot size, minimum investment, several
  ratios), pick only the 2-3 most important/striking ones to actually narrate in
  detail and leave the rest out, rather than compressing every available number
  into the script just because it was given. Whenever narration states a ratio or
  technical term a retail viewer may not know (e.g. ROE, ROCE, P/E, CAGR,
  EV/EBITDA, offer-for-sale), immediately follow it with a short plain-English
  clause explaining what it practically means (e.g. a high ROE -- meaning it
  generates a lot of profit from shareholder money), not just the bare number.
- Keep total narration length close to the requested duration at ~140 words/minute.
- visual_query must describe concrete, filmable subjects (avoid abstract concepts). If
  the topic names a specific company/industry, prefer a term tied to that company's
  actual business or sector (e.g. "gold jewellery workshop" for a jewellery company,
  "cement factory kiln" for a cement company) over generic finance stock imagery
  ("stock market chart", "money growth", "businessman handshake") -- those generic
  terms are what every finance channel's footage already looks like, so only fall
  back to them when the topic genuinely gives no sector/subject detail to search for.
- on_screen_text should surface the key number/name/fact from that scene's narration
  as short punchy text (e.g. "Revenue: Rs 412 Cr", "P/E: 22.5", "Roman Aqueducts, 312 BC"),
  so a muted viewer can still follow along. Use it whenever the scene has a concrete
  fact, figure, name, or date. Leave it as an empty string only for pure intro/outro/
  transition scenes with nothing concrete to show.
- The very first scene's on_screen_text MUST be the name of the main subject of the
  video (e.g. the company name, the person's name, the place/topic name) if the user's
  prompt names one — always show it, even though scene 1 is the hook/intro.
- diagram: prefer setting this whenever a scene's concept is a numeric comparison,
  trend, or sequence (e.g. comparing two companies' P/E ratios, a revenue trend over
  time) -- even when a filmable stock-footage equivalent also exists. A diagram built
  from this topic's own real numbers is unique to this video; generic stock footage
  of a "stock chart" or "office building" looks the same on every finance channel.
  Use ONLY numbers that literally appear in this scene's own narration — never invent
  new ones. Must be one of:
    {"kind": "number_line", "min": <int>, "max": <int>, "highlight": [<int>, ...]}
    {"kind": "triangular_dots", "rows": <int>, "label": "<short string>"}
    {"kind": "sequence_boxes", "items": ["<str>", ...], "highlight_index": <int or null>}
    {"kind": "bar_compare", "labels": ["<str>", ...], "values": [<number>, ...], "unit": "<str>"}
    {"kind": "shape_progression", "shape": "<str>", "counts": [<int>, ...]}
    {"kind": "trend_line", "labels": ["<str>", ...], "values": [<number>, ...], "unit": "<str>"}
    {"kind": "donut_share", "labels": ["<str>", ...], "values": [<number>, ...], "unit": "<str>"}
  Prefer number_line whenever the narration states a ratio/metric being
  judged against a threshold or healthy range (e.g. "debt-to-equity below 1",
  "P/E between 10 and 20") -- set min/max to a sensible range spanning that
  threshold and highlight the company's actual value, so a muted viewer can
  see at a glance where the real number falls, not just hear it read out.
  Prefer trend_line over bar_compare for a value changing across an ordered
  sequence (e.g. revenue/profit by year, a stock's price over time, compound
  growth) -- reserve bar_compare for comparing a handful of unordered items
  (e.g. two companies' P/E ratios side by side). Use donut_share only when the
  values are real parts of a single whole that sums to ~100% or to a stated
  total (e.g. revenue split by segment, holding composition) -- never for an
  unrelated comparison, which belongs in bar_compare instead.
  For bar_compare/trend_line/donut_share, "unit" must state what the values
  are in (e.g. "Cr", "%", "x") exactly as the narration states it -- a number
  shown on a diagram with no unit is as misleading as one with the wrong
  unit. Use "" only if the values are genuinely unitless (e.g. a count) --
  for donut_share, leave "unit" as "" when the values are plain percentages.
  Leave diagram as null for anything else, including every hook/intro/outro scene.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rstrip("`").strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)


def write_script_from_prompt(prompt: str, duration_sec: int = 60, extra_tags: list = None, style: str = None) -> dict:
    """If the prompt already lays out a timed, section-by-section script (e.g.
    "0-5 seconds - Hook", "5-15 seconds - Overview", ...), each section becomes
    its own scene via a narrow per-section LLM call, keeping narration, visual
    and on-screen text tightly bound to the same source facts. Otherwise falls
    back to a single freeform generation call for the whole prompt.

    extra_tags: tags that must appear regardless of what the LLM comes up with
    (e.g. "NCERT", "CBSE" for a school-syllabus channel) — see
    segmented_script.merge_tags.

    style: overall tone/style guidance for the video. Threaded into both paths
    so a segmented (structured) prompt gets the same style guidance a freeform
    one does, instead of silently dropping it.
    """
    segments = segmented_script.parse_segments(prompt)
    if segments:
        script = segmented_script.write_segmented_script(prompt, segments, duration_sec, extra_tags, global_instructions=style)
    else:
        freeform_prompt = f"{prompt}\n\nStyle/tone: {style}" if style else prompt
        script = _write_freeform_script(freeform_prompt, duration_sec, extra_tags)
    if script.get("symbol") and segmented_script.QUOTE_RE.search(prompt):
        # Hand-written quoted lines are spoken word for word -- compare their
        # figures with Screener before the video is made (prompt_factcheck).
        data = fundamentals.fetch_fundamentals(script["symbol"])
        script["_fundamentals"] = data
        script["factcheck"] = prompt_factcheck.check(prompt, data)
        for warning in script["factcheck"]:
            print(f"[factcheck] WARNING: {warning}")
    if not script.get("symbol"):
        # Listed stocks get their thumbnail from Screener (producer); an IPO
        # takes its numbers from this prompt instead.
        script["company_name"] = script.get("company_name") or segmented_script._ipo_company_from_heading(prompt)
        ipo_spec = thumbnail_spec.build_ipo(prompt, script.get("company_name"), script.get("title", ""))
        if ipo_spec:
            script["thumbnail_spec"] = ipo_spec
    return script


def _write_freeform_script(prompt: str, duration_sec: int = 60, extra_tags: list = None) -> dict:
    target_words = int(duration_sec / 60 * 140)
    # write_script_from_prompt appends "\n\nStyle/tone: ..." to the prompt; that
    # label's capitalized "Style" otherwise gets picked as a company-name
    # candidate for any theme that names no company, and Screener's search for
    # "Style" resolves to "Baazar Style Retail" -- the same company, every time
    # (observed in production: 4 different generic themes -> 4 near-identical
    # Baazar Style Retail videos). Ground/resolve on the user's own text only.
    grounding_text = prompt.split("\n\nStyle/tone:")[0]
    is_finance_theme = research.needs_grounding(grounding_text)

    user_prompt = f"{prompt.strip()}\n\n"

    symbol = research.resolve_symbol_for_theme(grounding_text) if is_finance_theme else None
    resolved_company = research.resolve_company_for_theme(grounding_text) if is_finance_theme else {}
    if is_finance_theme:
        facts = research.fetch_grounding_facts(grounding_text, symbol=symbol)
        if facts:
            user_prompt += (
                "Below are real, sourced facts from a live web search -- use ONLY "
                "these specific numbers, names, and claims; do not invent or add any "
                "other specific figures:\n"
                f"{facts}\n\n"
            )
        else:
            user_prompt += (
                "No verified real-world data could be retrieved for this topic. Do NOT "
                "invent specific numbers, prices, percentages, or dates -- keep the "
                "narration general and educational instead.\n\n"
            )
        if resolved_company.get("name"):
            # The theme itself only named a sector/topic (e.g. "Semiconductor
            # Stocks: Opportunity and Risk Analysis"), not a specific company --
            # the facts above are all about ONE real, verified company resolved
            # behind the scenes. Without telling the model which one, it has no
            # way to name it in the title/narration (observed in production: a
            # published video that never named its subject at all).
            user_prompt += (
                f"This video's facts above are specifically about {resolved_company['name']} "
                f"(NSE/BSE: {resolved_company['symbol']}) -- always name this exact company in "
                "the title, narration, and on-screen text; never leave it unnamed or refer to it "
                "only generically (e.g. \"this stock\"), and never speak the ticker aloud.\n\n"
            )

    dyk_beat = _find_did_you_know(resolved_company, symbol) if is_finance_theme else {"facts": [], "narration": ""}
    dyk_facts = dyk_beat["facts"]
    dyk_words = len(_did_you_know_narration(dyk_beat).split()) if dyk_facts else 0

    user_prompt += (
        f"Target narration length: about {max(20, target_words - dyk_words)} words "
        f"(~{duration_sec} seconds spoken).\n\n"
    )
    if dyk_facts:
        user_prompt += (
            "A separate \"Did you know?\" scene is inserted automatically right after scene 1 -- "
            "do not write a \"did you know\" fact or company-history trivia scene yourself.\n\n"
        )
    user_prompt += (
        "Scene 1 (the hook) must open its first sentence with the single most striking verified "
        "number from the content above, within its first 6-8 words -- never with the company name, a "
        "date, or \"In this video\" / other background framing before that number.\n\n"
    )
    if is_finance_theme:
        user_prompt += (
            "In the last content scene (just before any subscribe ask), end with one short yes-or-no "
            "question inviting viewers to answer in the comments (e.g. whether the valuation already "
            "prices in the growth) -- a question only, never a recommendation.\n\n"
        )
    style_guidance = youtube_insights.get_combined_guidance()
    if style_guidance:
        user_prompt += (
            "Below is guidance on what's currently drawing viewers — based on this "
            "channel's own data and/or the wider niche — apply it to the title and the "
            "first scene's hook, without contradicting the facts above. The title rules "
            "in your instructions always take priority over this guidance:\n"
            f"{style_guidance}\n\n"
        )
    user_prompt += "Produce the JSON script now."

    last_error = None
    for attempt in range(3):
        try:
            raw = llm.chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=4096, timeout=120)
            script = _extract_json(raw)
            for key in ("title", "description", "tags", "scenes"):
                if key not in script:
                    raise ValueError(f"Script missing required field: {key}")
            if not script["scenes"]:
                raise ValueError("Script has zero scenes")
            if not all(isinstance(scene, dict) for scene in script["scenes"]):
                # Observed in production: the model occasionally collapses a
                # scene to a bare narration string instead of an object (e.g.
                # "scenes": ["Trent Ltd posted...", {...}]) -- scene.get(...)
                # below then raises AttributeError, which this loop's
                # (ValueError, KeyError) catch doesn't cover, crashing the
                # whole run instead of retrying like every other malformed-
                # response case. Raise ValueError here so it retries too.
                raise ValueError("Script scenes must all be JSON objects")
            script["title"] = segmented_script.clean_title(script["title"])
            if is_finance_theme:
                script["title"] = segmented_script.strip_title_emoji(script["title"])
            script["description"] = segmented_script.normalize_indian_currency_units(script["description"])
            script["tags"] = segmented_script.merge_tags(extra_tags or [], script["tags"] or ["short video"])
            for scene in script["scenes"]:
                scene["narration"] = segmented_script.normalize_indian_currency_units(scene.get("narration") or "")
                diagram = scene.get("diagram")
                if diagram:
                    ok, _reason = diagrams.validate_diagram_spec(diagram, source_text=scene.get("narration", ""))
                    if not ok:
                        scene["diagram"] = None
            segmented_script.append_subscribe_cta(script["scenes"], ipo=segmented_script.is_ipo_theme(grounding_text))
            for i, scene in enumerate(script["scenes"]):
                _ground_scene_on_screen_text(scene, is_first_scene=(i == 0))
            if dyk_facts:
                script["scenes"].insert(1, _did_you_know_scene(dyk_beat))
            script["company_name"] = resolved_company.get("name") or _extract_freeform_company_name(script, is_finance_theme)
            if resolved_company.get("name"):
                # The theme never named a company for the model to echo into
                # on_screen_text in the first place (see the comment on
                # resolved_company above) -- SYSTEM_PROMPT's "first scene must
                # show the subject name" rule has nothing to work from without
                # this, so set it directly from the resolved real name instead.
                script["scenes"][0]["on_screen_text"] = resolved_company["name"]
            if script["company_name"]:
                # Deterministic backstop -- same reasoning as
                # segmented_script.polish_stock_title: the prompt
                # instruction above isn't always enough on its own, so force
                # the resolved company's real name into the title rather than
                # trust the model to have included it, and strip the filler
                # phrases the title rules forbid. Insert it BEFORE the
                # one-and-only truncation below -- truncating first and then
                # prepending a name afterwards (the prior bug) silently pushed
                # the title back over the limit and cut it a second time with
                # no indication it was shortened (observed in production: a
                # title truncated mid-clause with no ellipsis).
                script["title"] = segmented_script.polish_stock_title(script["title"], script["company_name"])
            script["title"] = segmented_script._truncate_title(script["title"])
            if script["company_name"] and symbol:
                for scene in script["scenes"]:
                    scene["narration"] = segmented_script.despeak_ticker(scene["narration"], symbol, script["company_name"])
            return script
        except (ValueError, KeyError) as e:
            last_error = e
            continue
    raise RuntimeError(
        f"Model returned an invalid script 3 times in a row ({last_error}). Try again."
    )


def _find_did_you_know(resolved_company: dict, symbol) -> dict:
    """The "did you know" beat for the video's company (see
    company_insights.did_you_know_beat): {"facts": [...], "narration": "..."},
    facts empty when nothing verified -- best-effort, never raises. The company
    name comes from the Screener resolution, or from a symbol-to-name lookup
    when the theme carried an explicit "NSE: SYMBOL" marker instead."""
    empty = {"facts": [], "narration": ""}
    try:
        name = resolved_company.get("name")
        if not name and symbol:
            match = fundamentals.search_company(symbol)
            if (match.get("symbol") or "").upper() == symbol.upper():
                name = match.get("name")
        if not (name and symbol):
            return empty
        return company_insights.did_you_know_beat(name, fundamentals.fetch_fundamentals(symbol))
    except Exception as e:
        print(f"[insights] did-you-know lookup failed: {type(e).__name__}: {e}")
        return empty


def _did_you_know_narration(beat: dict) -> str:
    """The LLM-polished (and validated) narration when there is one, else the
    raw verified facts joined."""
    text = beat.get("narration") or " ".join(beat.get("facts") or [])
    return segmented_script.normalize_indian_currency_units("Did you know? " + text)


def _did_you_know_scene(beat: dict) -> dict:
    """A scene built only from already-verified fact text (or its validated
    rewording) -- no unchecked model output reaches it."""
    return {
        "narration": _did_you_know_narration(beat),
        "visual_query": segmented_script._fallback_visual("did you know"),
        "on_screen_text": "DID YOU KNOW?",
        "diagram": None,
    }


def _extract_freeform_company_name(script: dict, is_finance_theme: bool):
    """The freeform path has no per-section "content" field to run
    segmented_script._extract_company_name's "(NSE: SYMBOL)" regex against --
    most freeform finance themes are a plain one-line string (e.g. "National
    Aluminium Company - Stock Analysis For Long Term Growth") with no such
    marker. Without this, producer.py never had a company_name to pass to
    ai_thumbnail.generate_hook_background(), so the AI thumbnail silently
    never activated for ANY freeform-path video (confirmed via the
    "[ai_thumbnail] no company name available" log line on a real run).

    SYSTEM_PROMPT already requires the first scene's on_screen_text to be the
    video's subject name for exactly this reason -- reuse it instead of
    re-deriving a name from the raw theme string with a second regex. Only
    trusted for themes research.needs_grounding() already flagged as
    finance/named-entity topics, so a generic creative theme never forces an
    irrelevant "stock market" thumbnail background."""
    if not is_finance_theme:
        return None
    first_text = (script["scenes"][0].get("on_screen_text") or "").strip()
    if not first_text or first_text in segmented_script.SUBSCRIBE_CTA_ON_SCREEN_TEXTS:
        return None
    return first_text


def _ground_scene_on_screen_text(scene: dict, is_first_scene: bool = False) -> None:
    """The freeform path writes a whole multi-scene script in a single LLM
    call -- unlike the segmented path (segmented_script._write_segment_scene),
    nothing previously checked that a scene's on_screen_text actually came
    from THAT scene's own narration. In practice the model sometimes drifts:
    showing a metric spoken in a different scene, or one it never said at all
    (reported: a real published video where on-screen metrics had "completely
    no correlation" with the narration). Reuses the same grounding/fact-
    extraction helpers the segmented path already relies on, applied against
    this scene's own narration (there's no separate source "content" field
    in the freeform path -- the narration itself is the only per-scene source
    of truth).

    is_first_scene: SYSTEM_PROMPT requires the very first scene's on_screen_text
    to be the video's subject name, not a metric -- unlike every other scene,
    where on_screen_text IS meant to surface a metric. So once that name has
    survived the grounding check, leave it alone: don't let a scene that
    happens to also narrate 2+ numbers (e.g. "current price of X and a market
    cap of Y") have its subject name clobbered by those numbers."""
    narration = scene.get("narration") or ""
    on_screen_text = segmented_script.normalize_indian_currency_units((scene.get("on_screen_text") or "").strip())
    if on_screen_text and not segmented_script._is_grounded(on_screen_text, narration):
        on_screen_text = ""
    if is_first_scene and on_screen_text:
        scene["on_screen_text"] = on_screen_text
        return
    facts = segmented_script._extract_facts(narration)
    if len(facts) > 1:
        on_screen_text = " | ".join(facts)
    elif not on_screen_text and facts:
        on_screen_text = facts[0]
    scene["on_screen_text"] = on_screen_text
