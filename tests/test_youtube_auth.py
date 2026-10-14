from unittest import mock

from nacl import encoding, public

from scripts import youtube_auth


def _fake_response(json_data=None, raise_for_status=None):
    resp = mock.Mock()
    resp.json.return_value = json_data or {}
    resp.raise_for_status.side_effect = raise_for_status
    return resp


def test_push_to_github_secret_calls_github_api_with_encrypted_value(monkeypatch):
    monkeypatch.setattr(youtube_auth.getpass, "getpass", lambda prompt: "fake-pat")

    private_key = public.PrivateKey.generate()
    public_key_b64 = private_key.public_key.encode(encoding.Base64Encoder).decode("utf-8")

    calls = []

    def fake_get(url, headers):
        calls.append(("GET", url, headers))
        if url.endswith("/repos/arunrajramkumar/genyt"):
            return _fake_response({"id": 12345})
        if url.endswith("/environments/education/secrets/public-key"):
            return _fake_response({"key": public_key_b64, "key_id": "key-1"})
        raise AssertionError(f"unexpected GET {url}")

    put_calls = []

    def fake_put(url, headers, json):
        put_calls.append((url, headers, json))
        return _fake_response()

    monkeypatch.setattr(youtube_auth.requests, "get", fake_get)
    monkeypatch.setattr(youtube_auth.requests, "put", fake_put)

    youtube_auth._push_to_github_secret("education", '{"token": "abc"}')

    assert len(put_calls) == 1
    url, headers, body = put_calls[0]
    assert url == "https://api.github.com/repositories/12345/environments/education/secrets/YOUTUBE_TOKEN_JSON"
    assert headers["Authorization"] == "Bearer fake-pat"
    assert body["key_id"] == "key-1"

    sealed_box = public.SealedBox(private_key)
    import base64
    decrypted = sealed_box.decrypt(base64.b64decode(body["encrypted_value"]))
    assert decrypted == b'{"token": "abc"}'


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
