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

This tool does **not** upload to YouTube — you review and publish the output yourself.

## One-time setup

```bash
pip3 install -r requirements.txt
```

Edit `.env` and fill in:
- `PEXELS_API_KEY` — free, from https://www.pexels.com/api/
- `GROQ_API_KEY` — free, from https://console.groq.com/keys (used to write scripts)

ffmpeg is already installed for this machine.

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
- No upload automation is included by design (per your choice) — you stay in
  control of what actually gets published to your channel.
