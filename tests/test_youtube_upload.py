from unittest import mock

import pytest

from pipeline import config, youtube_upload


def test_load_credentials_raises_not_configured_when_no_token_file():
    with pytest.raises(youtube_upload.NotConfiguredError):
        youtube_upload._load_credentials()


def test_load_credentials_refreshes_expired_token(monkeypatch):
    token_path = config.CACHE_DIR / "youtube_token.json"
    token_path.write_text("{}")

    fake_creds = mock.Mock(expired=True, refresh_token="r")
    fake_creds.to_json.return_value = '{"refreshed": true}'
    monkeypatch.setattr(youtube_upload.Credentials, "from_authorized_user_file", lambda *a, **k: fake_creds)
    monkeypatch.setattr(youtube_upload, "Request", lambda: mock.Mock())

    creds = youtube_upload._load_credentials()

    assert creds is fake_creds
    fake_creds.refresh.assert_called_once()
    assert token_path.read_text() == '{"refreshed": true}'


class _FakeRequest:
    """Stand-in for a googleapiclient resumable MediaFileUpload request —
    resolves on the first next_chunk() call so tests don't need to simulate
    real chunked upload progress."""

    def __init__(self, response):
        self._response = response

    def next_chunk(self):
        return None, self._response


def test_upload_video_returns_url_and_uploads_captions(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    srt_path = tmp_path / "video.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")

    captured = {}

    def fake_videos_insert(part, body, media_body):
        captured["video_body"] = body
        return _FakeRequest({"id": "abc123"})

    captions_execute = mock.Mock(return_value={})

    def fake_captions_insert(part, body, media_body):
        captured["captions_body"] = body
        return mock.Mock(execute=captions_execute)

    fake_youtube = mock.Mock()
    fake_youtube.videos.return_value.insert.side_effect = fake_videos_insert
    fake_youtube.captions.return_value.insert.side_effect = fake_captions_insert
    monkeypatch.setattr(youtube_upload, "_client", lambda: fake_youtube)

    result = youtube_upload.upload_video(
        video_path, "Title", "Description", ["tag1", "tag2"],
        captions_path=srt_path, captions_language="ta",
    )

    assert result == {"video_id": "abc123", "url": "https://youtu.be/abc123"}
    assert captured["video_body"]["status"]["privacyStatus"] == config.YOUTUBE_DEFAULT_PRIVACY
    assert captured["video_body"]["snippet"]["title"] == "Title"
    assert captured["captions_body"]["snippet"]["language"] == "ta"
    captions_execute.assert_called_once()


def test_upload_video_respects_explicit_privacy_status(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")

    captured = {}

    def fake_videos_insert(part, body, media_body):
        captured["body"] = body
        return _FakeRequest({"id": "xyz"})

    fake_youtube = mock.Mock()
    fake_youtube.videos.return_value.insert.side_effect = fake_videos_insert
    monkeypatch.setattr(youtube_upload, "_client", lambda: fake_youtube)

    youtube_upload.upload_video(video_path, "T", "D", [], privacy_status="unlisted")

    assert captured["body"]["status"]["privacyStatus"] == "unlisted"
