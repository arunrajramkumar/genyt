# ytagent — fully-automated YouTube video generator

Give it a theme, it produces a finished, ready-to-upload video:

```
theme  -->  Groq-hosted LLM writes script  -->  Microsoft Edge TTS narrates each scene
       -->  Pexels fetches matching stock footage/photos
       -->  ffmpeg assembles everything into one MP4
```

Output per video (in `output/`):
- `<slug>.mp4` — the finished video (1920x1080, H.264 + AAC)
- `<slug>.srt` — captions file (upload to YouTube separately as closed captions —
  this ffmpeg build has no subtitle-burn-in support, see Notes below)
- `<slug>.metadata.txt` — title, description, tags to paste into YouTube Studio

Once a video finishes, it's automatically uploaded to your YouTube channel as
**private** (see "YouTube upload" below) — you still review and flip it to
public yourself, nothing goes live without you.

## One-time setup

```bash
pip3 install -r requirements.txt
```

Edit `.env` and fill in:
- `PEXELS_API_KEY` — free, from https://www.pexels.com/api/
- `GROQ_API_KEY` — free, from https://console.groq.com/keys (used to write scripts)

ffmpeg is already installed for this machine.

### YouTube upload

Videos are uploaded automatically (as **private**) once generated. One-time setup:

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project
   (or reuse one), enable the **YouTube Data API v3**, then create an OAuth
   client under Credentials — type **Desktop app**.
2. Download its JSON and save it as `youtube_client_secret.json` in the repo
   root (gitignored — never commit it).
3. Run `python3 scripts/youtube_auth.py` once. It opens a browser — sign in
   with the Google account that manages the target channel and grant access.
   This saves a reusable token to `cache/youtube_token.json` (also gitignored).
4. Re-run step 3 any time `cache/` is wiped or you switch channels.

Until step 3 is done, uploads are silently skipped (video/captions/metadata
still land in `output/` as usual) — generation never fails because of a
missing YouTube token.

Config knobs in `.env`:
- `YOUTUBE_AUTO_UPLOAD=false` — disable automatic upload entirely (local-only mode)
- `YOUTUBE_DEFAULT_PRIVACY=private|unlisted|public` — default `private`
- `YOUTUBE_CLIENT_SECRETS_FILE=/path/to/file.json` — override the default repo-root path

## Usage

Edit `themes.yaml` to add topics you want videos for, then:

```bash
python3 run.py            # produce a video for the next un-done theme
python3 run.py --all      # produce videos for every un-done theme
python3 run.py --theme "3 weird facts about octopuses" --duration 120   # one-off, skip themes.yaml
```

Each theme in `themes.yaml` is marked `done: true` after its video is produced, so
you can re-run `run.py` on a schedule (e.g. via `cron`) and it will only ever pick
up new themes you add.

### Running on a schedule (cron)

```
# every day at 9am, generate one video from themes.yaml
0 9 * * * cd /Users/aramkumar/ytagent && /usr/bin/python3 run.py >> cron.log 2>&1
```

### Generating from your phone via Telegram

Instead of the web UI, you can message a Telegram bot with a prompt and get the
finished video sent straight back to the chat:

1. In Telegram, message **@BotFather** → `/newbot` → follow the prompts → copy
   the bot token it gives you.
2. Message your new bot once (anything), then visit
   `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser to find your
   numeric `chat.id` — this is your Telegram user ID.
3. Set these environment variables on your deployment (e.g. Render's
   dashboard → Environment):
   - `TELEGRAM_BOT_TOKEN` — from step 1.
   - `TELEGRAM_WEBHOOK_SECRET` — any random string you make up.
   - `TELEGRAM_ALLOWED_CHAT_IDS` — your chat ID from step 2 (comma-separate for
     more than one person). Leave unset to allow anyone who finds the bot —
     not recommended.
4. After deploying, register the webhook once (replace placeholders):
   ```bash
   curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
     -d "url=https://<your-app>.onrender.com/telegram-webhook" \
     -d "secret_token=<TELEGRAM_WEBHOOK_SECRET>"
   ```
5. Message your bot a prompt (e.g. "5 facts about octopuses") — it replies with
   the video, thumbnail, captions, and metadata once generation finishes.
6. By default narration uses `TTS_VOICE` (Indian English). To switch languages
   per-chat, send `/voice` to see supported languages, then e.g. `/voice hindi`
   or `/voice tamil` — it applies to every video you request afterward until
   changed again. Advanced: send any exact edge-tts voice ID (e.g.
   `/voice es-MX-DaliaNeural`) for finer control — run `edge-tts --list-voices`
   for the full catalog.

## Testing

This repo has a pytest suite covering every module (script generation/grounding
rules, font rendering, voice selection, Telegram bot commands, ffmpeg command
construction, the web UI's auth/routing). It's fully mocked — no network calls,
no real ffmpeg/TTS — so the whole thing runs in a couple of seconds.

**Run the tests before every commit — this is mandatory, not optional**, so a
new feature can't silently break an existing one (e.g. the font-rendering
change and the Groq reasoning_effort fix were both caught by re-running this
suite after unrelated work).

```bash
pip3 install -r requirements-dev.txt
./scripts/run_tests.sh                 # fast suite — run this before every commit
./scripts/run_tests.sh --integration   # + slower real-ffmpeg checks (run when touching pipeline/assemble.py)
```

To make this automatic, enable the repo's pre-commit hook once per clone:

```bash
git config core.hooksPath .githooks
```

With the hook enabled, `git commit` refuses to proceed if any test fails.
When you add a new feature, add or update tests for it in `tests/` in the same
commit — don't let coverage drift behind the code.

## Notes / limitations

- **Captions aren't burned into the video.** This machine's ffmpeg was built from
  source (non-standard Homebrew prefix) without `libass`, so the `subtitles` filter
  isn't available. The `.srt` file is generated separately — upload it in YouTube
  Studio under Subtitles for the same effect, with the benefit that viewers can
  toggle it off.
- **Script quality** depends on the Groq model in use. Swap models via
  `GROQ_MODEL` in `.env` — see https://console.groq.com/docs/models for the
  current free-tier catalog.
- **Stock visuals** come from Pexels based on the model's `visual_query` per
  scene. Quality depends on how well the query matches available stock footage —
  review output before publishing.
- **YouTube upload** happens automatically but defaults to private — see
  "YouTube upload" above for one-time OAuth setup. You still control when
  (and whether) a video actually goes public.
