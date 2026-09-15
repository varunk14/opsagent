"""
Langfuse on this machine: compose.tracing.yml, and the .env that feeds it.

The stack is its own compose file rather than a profile of docker-compose.yml: compose
checks every required variable in a file, even for services it will not start, so a
profile would stop `docker compose up db` working until Langfuse's secrets existed.

Nothing in it is reachable from off this machine: only the web UI is published, and
only on 127.0.0.1; its databases publish nothing. Usage telemetry is off. Every image
is pinned to a digest. No secret has a default -- a missing one stops compose -- and no
login is seeded: whoever wants the UI creates one.

`python -m app.tracing env` writes those secrets, fresh and random, together with the
settings a worker and the screen need to use them. It adds only what is missing, never
changes a line already there, keeps the file private, and prints no secret.
"""

import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from app.tracing import exporter_from_env, main, write_env
from app.web import langfuse_link_base

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "compose.tracing.yml"
DATABASES = ("postgres", "redis", "clickhouse", "minio")
REQUIRED = re.compile(r"\$\{([A-Z0-9_]+):\?\}")
WORKER_SETTINGS = (
    "OPSAGENT_OTLP_ENDPOINT",
    "OPSAGENT_LANGFUSE_PUBLIC_KEY",
    "OPSAGENT_LANGFUSE_SECRET_KEY",
    "OPSAGENT_LANGFUSE_PROJECT_URL",
)
SECRETS = (
    "LANGFUSE_POSTGRES_PASSWORD",
    "LANGFUSE_CLICKHOUSE_PASSWORD",
    "LANGFUSE_MINIO_PASSWORD",
    "LANGFUSE_REDIS_PASSWORD",
    "LANGFUSE_NEXTAUTH_SECRET",
    "LANGFUSE_SALT",
    "LANGFUSE_ENCRYPTION_KEY",
    "LANGFUSE_SECRET_KEY",
    "OPSAGENT_LANGFUSE_SECRET_KEY",
)


def services() -> dict:
    return yaml.safe_load(COMPOSE.read_text())["services"]


def settings(path: Path) -> dict[str, str]:
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return {name: value for name, value in pairs}


# --- the stack ------------------------------------------------------------------------------


def test_only_the_web_ui_is_published_and_only_on_this_machine():
    published = {name: service.get("ports", []) for name, service in services().items() if service.get("ports")}

    assert published == {"langfuse-web": ["127.0.0.1:3000:3000"]}


def test_the_stacks_databases_publish_nothing():
    for name in DATABASES:
        assert "ports" not in services()[name], name


def test_usage_telemetry_is_off_for_langfuse():
    for name in ("langfuse-web", "langfuse-worker"):
        assert services()[name]["environment"]["TELEMETRY_ENABLED"] == "false", name


def test_every_image_is_pinned_to_a_digest():
    for name, service in services().items():
        assert re.search(r"@sha256:[0-9a-f]{64}$", service["image"]), name


def test_no_secret_has_a_default_value():
    text = COMPOSE.read_text()
    every_variable = re.findall(r"\$\{[^}]*\}", text)

    assert every_variable, "the stack takes its secrets from .env"
    assert all(REQUIRED.fullmatch(variable) for variable in every_variable), every_variable
    assert set(REQUIRED.findall(text)) == {name for name in SECRETS if name.startswith("LANGFUSE_")} | {
        "LANGFUSE_PUBLIC_KEY"
    }


def test_no_login_is_seeded():
    for name, service in services().items():
        environment = service.get("environment", {})
        assert not [key for key in environment if key.startswith("LANGFUSE_INIT_USER")], name


def test_the_main_stack_needs_none_of_langfuses_settings():
    assert "${" not in (ROOT / "docker-compose.yml").read_text()


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")
def test_compose_accepts_the_stack_with_a_generated_env_and_refuses_it_without(tmp_path):
    env = tmp_path / ".env"
    write_env(env)
    empty = tmp_path / "empty.env"
    empty.write_text("")

    def config(env_file: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE), "--env-file", str(env_file), "config", "--quiet"],
            capture_output=True, text=True, timeout=60, check=False,
        )

    accepted = config(env)
    assert accepted.returncode == 0, accepted.stderr
    refused = config(empty)
    assert refused.returncode != 0
    assert "required variable" in refused.stderr


# --- the .env -------------------------------------------------------------------------------


def test_the_env_holds_every_secret_the_stack_requires_and_is_private(tmp_path):
    env = tmp_path / ".env"

    write_env(env)

    values = settings(env)
    assert set(REQUIRED.findall(COMPOSE.read_text())) <= set(values)
    assert set(WORKER_SETTINGS) <= set(values)
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_the_worker_and_screen_settings_point_at_the_stack(tmp_path):
    env = tmp_path / ".env"

    write_env(env)

    values = settings(env)
    assert values["OPSAGENT_LANGFUSE_PUBLIC_KEY"] == values["LANGFUSE_PUBLIC_KEY"]
    assert values["OPSAGENT_LANGFUSE_SECRET_KEY"] == values["LANGFUSE_SECRET_KEY"]
    assert values["OPSAGENT_OTLP_ENDPOINT"] == "http://127.0.0.1:3000/api/public/otel"
    assert exporter_from_env(values) is not None
    assert langfuse_link_base(values["OPSAGENT_LANGFUSE_PROJECT_URL"]) == "http://127.0.0.1:3000/project/opsagent-local"


def test_the_keys_have_langfuses_shape_and_the_encryption_key_is_256_bits(tmp_path):
    env = tmp_path / ".env"

    write_env(env)

    values = settings(env)
    assert re.fullmatch(r"pk-lf-[0-9a-f-]{36}", values["LANGFUSE_PUBLIC_KEY"])
    assert re.fullmatch(r"sk-lf-[0-9a-f-]{36}", values["LANGFUSE_SECRET_KEY"])
    assert re.fullmatch(r"[0-9a-f]{64}", values["LANGFUSE_ENCRYPTION_KEY"])


def test_every_value_is_safe_to_source_from_a_shell(tmp_path):
    env = tmp_path / ".env"

    write_env(env)

    for name, value in settings(env).items():
        assert re.fullmatch(r"[A-Za-z0-9._:/-]+", value), name


def test_secrets_are_fresh_each_time(tmp_path):
    first, second = tmp_path / "first.env", tmp_path / "second.env"

    write_env(first)
    write_env(second)

    for name in SECRETS:
        assert settings(first)[name] != settings(second)[name], name


def test_settings_already_in_the_file_are_never_changed(tmp_path):
    env = tmp_path / ".env"
    before = "OPSAGENT_OPERATOR=asha\nLANGFUSE_SALT=mine\nLANGFUSE_PUBLIC_KEY=pk-lf-mine\n"
    env.write_text(before)

    added = write_env(env)

    text = env.read_text()
    assert text.startswith(before)
    assert text.count("LANGFUSE_SALT=") == 1
    assert "LANGFUSE_SALT" not in added
    assert settings(env)["OPSAGENT_LANGFUSE_PUBLIC_KEY"] == "pk-lf-mine"
    assert write_env(env) == []
    assert env.read_text() == text


def test_the_command_names_what_it_added_and_prints_no_secret(tmp_path, capsys):
    env = tmp_path / ".env"

    assert main(["app.tracing", "env", "--path", str(env)]) == 0

    out = capsys.readouterr().out
    values = settings(env)
    assert str(env) in out
    assert values["LANGFUSE_PUBLIC_KEY"] in out
    for name in SECRETS:
        assert values[name] not in out, name


def test_an_existing_file_others_could_read_is_made_private_before_secrets_go_in(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPSAGENT_OPERATOR=asha\n")
    env.chmod(0o644)

    write_env(env)

    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_a_symbolic_link_in_place_of_the_file_is_refused_and_its_target_untouched(tmp_path):
    target = tmp_path / "somewhere-else"
    target.write_text("kept as it was\n")
    env = tmp_path / ".env"
    env.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        write_env(env)

    assert target.read_text() == "kept as it was\n"


def test_the_command_reports_a_refused_file_without_a_traceback(tmp_path, capsys):
    target = tmp_path / "somewhere-else"
    target.write_text("")
    env = tmp_path / ".env"
    env.symlink_to(target)

    assert main(["app.tracing", "env", "--path", str(env)]) == 2

    assert "symbolic link" in capsys.readouterr().err
    assert target.read_text() == ""


def test_running_again_makes_a_file_others_could_read_private_even_with_nothing_to_add(tmp_path):
    env = tmp_path / ".env"
    write_env(env)
    env.chmod(0o644)
    before = env.read_text()

    assert write_env(env) == []

    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert env.read_text() == before


def test_a_last_line_without_a_newline_is_kept_whole(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPSAGENT_OPERATOR=asha")

    write_env(env)

    assert env.read_text().startswith("OPSAGENT_OPERATOR=asha\nLANGFUSE_")
    assert settings(env)["OPSAGENT_OPERATOR"] == "asha"


def test_comments_blanks_and_stray_words_set_nothing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# LANGFUSE_SALT=old\n#LANGFUSE_PUBLIC_KEY=pk-lf-old\n\njust some words\n")

    added = write_env(env)

    assert {"LANGFUSE_SALT", "LANGFUSE_PUBLIC_KEY"} <= set(added)
    assert settings(env)["OPSAGENT_LANGFUSE_PUBLIC_KEY"] != "pk-lf-old"


def test_exported_and_quoted_settings_count_as_already_there(tmp_path):
    env = tmp_path / ".env"
    env.write_text("export LANGFUSE_SALT=mine\nLANGFUSE_PUBLIC_KEY=\"pk-lf-mine\"\nLANGFUSE_SECRET_KEY='sk-lf-mine'\n")

    added = write_env(env)

    text = env.read_text()
    assert "LANGFUSE_SALT" not in added
    assert text.count("LANGFUSE_SALT=") == 1
    assert settings(env)["OPSAGENT_LANGFUSE_PUBLIC_KEY"] == "pk-lf-mine"
    assert settings(env)["OPSAGENT_LANGFUSE_SECRET_KEY"] == "sk-lf-mine"


def test_the_command_run_again_changes_nothing_and_says_so(tmp_path, capsys):
    env = tmp_path / ".env"
    main(["app.tracing", "env", "--path", str(env)])
    before = env.read_text()
    capsys.readouterr()

    assert main(["app.tracing", "env", "--path", str(env)]) == 0

    out = capsys.readouterr().out
    assert "nothing changed" in out
    assert settings(env)["LANGFUSE_PUBLIC_KEY"] in out
    assert env.read_text() == before


def test_the_command_refuses_anything_but_env(capsys):
    with pytest.raises(SystemExit) as refused:
        main(["app.tracing", "nonsense"])

    assert refused.value.code != 0


# --- the secret scan ------------------------------------------------------------------------


def test_the_ci_secret_scan_catches_a_langfuse_secret_key(tmp_path):
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    (pattern,) = re.findall(r'grep -rnE "([^"]+)"', workflow)
    env = tmp_path / ".env"
    write_env(env)

    assert re.search(pattern, settings(env)["LANGFUSE_SECRET_KEY"])
