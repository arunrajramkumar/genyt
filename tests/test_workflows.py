import subprocess
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "auto-video.yml"


def _load():
    return yaml.safe_load(WORKFLOW.read_text())


def _run_step():
    steps = _load()["jobs"]["generate"]["steps"]
    return next(s for s in steps if s.get("name") == "Run pipeline")


def test_manual_inputs_include_prompt_file_and_keep_existing():
    inputs = _load()[True]["workflow_dispatch"]["inputs"]
    assert {"theme", "prompt_file", "duration"} <= set(inputs)
    assert inputs["prompt_file"]["required"] is False


def test_inputs_reach_the_script_via_env_not_interpolation():
    step = _run_step()
    assert "${{" not in step["run"]
    assert set(step["env"]) >= {"INPUT_THEME", "INPUT_PROMPT_FILE", "INPUT_DURATION"}


def test_scheduled_path_still_runs_trending():
    assert "python3 run.py --trending" in _run_step()["run"]


def _run_script(tmp_path, **env):
    """Executes the step's shell with `python3 run.py` replaced by an echo
    stub so we see exactly which branch/args it takes."""
    script = _run_step()["run"].replace("python3 run.py", "printf '%s\\n' ARGS")
    script = script.replace("ARGS", "run.py")
    full_env = {"PATH": "/usr/bin:/bin", "INPUT_THEME": "", "INPUT_PROMPT_FILE": "", "INPUT_DURATION": ""}
    full_env.update(env)
    return subprocess.run(["bash", "-c", script], cwd=tmp_path, env=full_env, capture_output=True, text=True)


def test_prompt_file_is_read_and_takes_priority(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "a.txt").write_text('0-5 seconds - Hook\nsay "hi" $HOME `x`\n')
    r = _run_script(tmp_path, INPUT_PROMPT_FILE="prompts/a.txt", INPUT_THEME="ignored")
    assert r.returncode == 0
    assert "0-5 seconds - Hook" in r.stdout and '$HOME `x`' in r.stdout
    assert "ignored" not in r.stdout


def test_missing_or_unsafe_prompt_file_fails(tmp_path):
    assert _run_script(tmp_path, INPUT_PROMPT_FILE="nope.txt").returncode == 1
    assert _run_script(tmp_path, INPUT_PROMPT_FILE="../x.txt").returncode == 1
    assert _run_script(tmp_path, INPUT_PROMPT_FILE="/etc/passwd").returncode == 1


def test_theme_with_shell_metacharacters_is_passed_literally(tmp_path):
    r = _run_script(tmp_path, INPUT_THEME='Foo "bar" $(touch pwned) `id`')
    assert r.returncode == 0
    assert 'Foo "bar" $(touch pwned) `id`' in r.stdout
    assert not (tmp_path / "pwned").exists()


def test_default_duration_is_60(tmp_path):
    assert "60" in _run_script(tmp_path, INPUT_THEME="x").stdout
    assert "90" in _run_script(tmp_path, INPUT_THEME="x", INPUT_DURATION="90").stdout
