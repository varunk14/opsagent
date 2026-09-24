"""
Where the screen binds, and which hosts it answers to, so it can sit behind a reverse proxy.

On this machine the screen binds loopback and answers only to localhost -- a safety, not an accident.
Behind Caddy on a real host it must bind every interface inside its container (reachable only over
the compose network, never published) and answer to the domain the proxy forwards. Both come from
the environment, and both default to the safe local values, so nothing changes for local use.
"""

from fastapi.testclient import TestClient

from app.web import allowed_hosts_from_env, create_app

DUMMY = "postgresql://opsagent:dev@localhost:5432/opsagent_test"


def test_it_answers_to_localhost_by_default():
    assert allowed_hosts_from_env({}) == ["127.0.0.1", "localhost"]


def test_the_allowed_hosts_come_from_the_environment():
    assert allowed_hosts_from_env({"OPSAGENT_ALLOWED_HOSTS": "opsagent.example.com"}) == [
        "opsagent.example.com"
    ]


def test_several_hosts_are_split_and_trimmed():
    assert allowed_hosts_from_env({"OPSAGENT_ALLOWED_HOSTS": "a.example.com, b.example.com"}) == [
        "a.example.com",
        "b.example.com",
    ]


def test_a_configured_host_is_answered():
    app = create_app(dsn=DUMMY, allowed_hosts=["opsagent.example.com"])
    client = TestClient(app, base_url="https://opsagent.example.com")
    # The front door renders the home page without touching the database, so this needs no schema.
    assert client.get("/", follow_redirects=False).status_code == 200


def test_another_host_is_refused():
    app = create_app(dsn=DUMMY, allowed_hosts=["opsagent.example.com"])
    client = TestClient(app, base_url="https://evil.example.com")
    assert client.get("/", follow_redirects=False).status_code == 400
