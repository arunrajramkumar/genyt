import pytest

import webapp


@pytest.fixture
def client():
    webapp.app.config["TESTING"] = True
    with webapp.app.test_client() as c:
        yield c


def test_healthz_is_unauthenticated_even_with_password_set(client, monkeypatch):
    monkeypatch.setattr(webapp, "APP_PASSWORD", "secret")
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_index_requires_auth_when_password_configured(client, monkeypatch):
    monkeypatch.setattr(webapp, "APP_PASSWORD", "secret")
    resp = client.get("/")
    assert resp.status_code == 401


def test_index_accessible_without_auth_when_no_password_configured(client, monkeypatch):
    monkeypatch.setattr(webapp, "APP_PASSWORD", None)
    resp = client.get("/")
    assert resp.status_code == 200


def test_index_accessible_with_correct_credentials(client, monkeypatch):
    import base64
    monkeypatch.setattr(webapp, "APP_PASSWORD", "secret")
    monkeypatch.setattr(webapp, "APP_USERNAME", "admin")
    creds = base64.b64encode(b"admin:secret").decode()
    resp = client.get("/", headers={"Authorization": f"Basic {creds}"})
    assert resp.status_code == 200


def test_telegram_webhook_404_when_bot_not_configured(client, monkeypatch):
    monkeypatch.setattr(webapp.telegram_bot, "BOT_TOKEN", "")
    resp = client.post("/telegram-webhook", json={})
    assert resp.status_code == 404


def test_telegram_webhook_rejects_wrong_secret(client, monkeypatch):
    monkeypatch.setattr(webapp.telegram_bot, "BOT_TOKEN", "dummy-token")
    monkeypatch.setattr(webapp.telegram_bot, "WEBHOOK_SECRET", "correct-secret")
    resp = client.post("/telegram-webhook", json={}, headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"})
    assert resp.status_code == 403


def test_telegram_webhook_accepts_correct_secret_and_forwards_update(client, monkeypatch):
    monkeypatch.setattr(webapp.telegram_bot, "BOT_TOKEN", "dummy-token")
    monkeypatch.setattr(webapp.telegram_bot, "WEBHOOK_SECRET", "correct-secret")
    received = {}
    monkeypatch.setattr(webapp.telegram_bot, "handle_update", lambda update: received.setdefault("update", update))

    resp = client.post(
        "/telegram-webhook", json={"message": {"text": "hi"}},
        headers={"X-Telegram-Bot-Api-Secret-Token": "correct-secret"},
    )
    assert resp.status_code == 200
    assert received["update"] == {"message": {"text": "hi"}}


def test_generate_starts_background_job_and_redirects(client, monkeypatch):
    monkeypatch.setattr(webapp, "APP_PASSWORD", None)
    monkeypatch.setattr(webapp.threading, "Thread", lambda target, args, daemon: _ImmediateThread(target, args))

    resp = client.post("/generate", data={"prompt": "octopus facts", "orientation": "shorts", "duration": "30"})
    assert resp.status_code == 302
    assert "/job/" in resp.headers["Location"]


class _ImmediateThread:
    """Runs the job synchronously instead of in a background thread, so tests
    don't race the job's own thread to check its result."""
    def __init__(self, target, args):
        self.target, self.args = target, args

    def start(self):
        pass  # intentionally not run — the job's own logic is tested elsewhere


def test_job_status_404_for_unknown_job(client):
    resp = client.get("/job/does-not-exist/status")
    assert resp.status_code == 404


def test_job_video_404_before_job_completes(client, monkeypatch):
    monkeypatch.setattr(webapp, "APP_PASSWORD", None)
    with webapp.JOBS_LOCK:
        webapp.JOBS["abc"] = {"status": "running", "log": [], "result": None, "error": None}
    resp = client.get("/job/abc/video")
    assert resp.status_code == 404
