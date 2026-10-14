from unittest import mock

from scripts import youtube_auth


def test_push_to_github_secret_calls_gh_with_correct_args(monkeypatch):
    calls = []
    monkeypatch.setattr(youtube_auth.subprocess, "run", lambda args, check: calls.append((args, check)))

    youtube_auth._push_to_github_secret("education", '{"token": "abc"}')

    args, check = calls[0]
    assert args == [
        "gh", "secret", "set", "YOUTUBE_TOKEN_JSON",
        "--repo", youtube_auth.GITHUB_REPO, "--env", "education", "--body", '{"token": "abc"}',
    ]
    assert check is True


def test_main_pushes_to_github_when_flag_passed(monkeypatch, tmp_path):
    monkeypatch.setattr(youtube_auth.sys, "argv", ["youtube_auth.py", "--push-to-github", "common"])
    monkeypatch.setattr(youtube_auth.config, "YOUTUBE_CLIENT_SECRETS_FILE", tmp_path / "secrets.json")
    (tmp_path / "secrets.json").write_text("{}")
    monkeypatch.setattr(youtube_auth.config, "CACHE_DIR", tmp_path)

    fake_creds = mock.Mock()
    fake_creds.to_json.return_value = '{"token": "xyz"}'
    fake_flow = mock.Mock()
    fake_flow.run_local_server.return_value = fake_creds
    monkeypatch.setattr(
        youtube_auth.InstalledAppFlow, "from_client_secrets_file", lambda path, scopes: fake_flow,
    )

    pushed = {}
    monkeypatch.setattr(youtube_auth, "_push_to_github_secret", lambda env, token: pushed.setdefault("call", (env, token)))

    youtube_auth.main()

    assert pushed["call"] == ("common", '{"token": "xyz"}')
    assert (tmp_path / "youtube_token.json").read_text() == '{"token": "xyz"}'


def test_main_skips_github_push_when_flag_not_passed(monkeypatch, tmp_path):
    monkeypatch.setattr(youtube_auth.sys, "argv", ["youtube_auth.py"])
    monkeypatch.setattr(youtube_auth.config, "YOUTUBE_CLIENT_SECRETS_FILE", tmp_path / "secrets.json")
    (tmp_path / "secrets.json").write_text("{}")
    monkeypatch.setattr(youtube_auth.config, "CACHE_DIR", tmp_path)

    fake_creds = mock.Mock()
    fake_creds.to_json.return_value = '{"token": "xyz"}'
    fake_flow = mock.Mock()
    fake_flow.run_local_server.return_value = fake_creds
    monkeypatch.setattr(
        youtube_auth.InstalledAppFlow, "from_client_secrets_file", lambda path, scopes: fake_flow,
    )

    def fail_if_called(env, token):
        raise AssertionError("should not push to GitHub when --push-to-github isn't passed")

    monkeypatch.setattr(youtube_auth, "_push_to_github_secret", fail_if_called)

    youtube_auth.main()
