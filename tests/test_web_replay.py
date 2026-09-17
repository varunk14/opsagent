"""
Replay from the screen: a button on a run, and the thread between a run and its replays.

The point of replay is comparison, so the screen has to make both sides reachable. A run carries a
button that queues a fresh run from the same message; the fresh run links back to the one it came
from; and the original lists the replays made of it, each a link to its own page and outcome. The
action is a POST carrying this screen's CSRF token, like every other decision here, so a page on
another site cannot trigger a replay.
"""

import re
from uuid import UUID

import psycopg
import pytest

from tests.test_approval_path import work
from tests.test_run_agent import happy_model, ledger, queue
from tests.test_web import client_for

pytestmark = pytest.mark.db


def token_on(client, path: str) -> str:
    page = client.get(path).text
    found = re.search(r'name="csrf" value="([^"]+)"', page)
    assert found, "the page carries no CSRF token"
    return found.group(1)


def worked_run(dsn: str) -> str:
    ledger(dsn)
    run_id = queue(dsn)
    work(dsn, happy_model())
    return run_id


def replays_of(dsn: str, run_id) -> list[str]:
    with psycopg.connect(dsn) as connection:
        return [
            str(row[0])
            for row in connection.execute(
                "SELECT id FROM runs WHERE replay_of = %s", (run_id,)
            ).fetchall()
        ]


# --- the button ---------------------------------------------------------------


def test_a_run_page_offers_to_replay_it(fresh_database, exported):
    run_id = worked_run(fresh_database)
    page = client_for(fresh_database).get(f"/runs/{run_id}").text
    assert f'action="/runs/{run_id}/replay"' in page


def test_replaying_from_the_screen_queues_a_linked_run(fresh_database, exported):
    run_id = worked_run(fresh_database)
    client = client_for(fresh_database)
    token = token_on(client, f"/runs/{run_id}")

    response = client.post(f"/runs/{run_id}/replay", data={"csrf": token}, follow_redirects=False)

    assert response.status_code == 303
    new_id = response.headers["location"].removeprefix("/runs/")
    assert UUID(new_id)  # a real id
    assert new_id != run_id
    assert replays_of(fresh_database, run_id) == [new_id]


# --- the thread between them --------------------------------------------------


def test_the_original_lists_its_replays(fresh_database, exported):
    run_id = worked_run(fresh_database)
    client = client_for(fresh_database)
    token = token_on(client, f"/runs/{run_id}")
    new_id = client.post(
        f"/runs/{run_id}/replay", data={"csrf": token}, follow_redirects=False
    ).headers["location"].removeprefix("/runs/")

    page = client.get(f"/runs/{run_id}").text
    assert f"/runs/{new_id}" in page


def test_a_replay_links_back_to_its_original(fresh_database, exported):
    run_id = worked_run(fresh_database)
    client = client_for(fresh_database)
    token = token_on(client, f"/runs/{run_id}")
    new_id = client.post(
        f"/runs/{run_id}/replay", data={"csrf": token}, follow_redirects=False
    ).headers["location"].removeprefix("/runs/")

    page = client.get(f"/runs/{new_id}").text
    assert f"/runs/{run_id}" in page


# --- it is a guarded action ---------------------------------------------------


@pytest.mark.parametrize("form", [{}, {"csrf": "guessed"}], ids=["no-token", "wrong-token"])
def test_a_replay_without_this_screens_token_is_refused(fresh_database, exported, form):
    run_id = worked_run(fresh_database)
    client = client_for(fresh_database)
    token_on(client, f"/runs/{run_id}")  # sets the cookie

    response = client.post(f"/runs/{run_id}/replay", data=form, follow_redirects=False)

    assert response.status_code == 403
    assert replays_of(fresh_database, run_id) == []


def test_replaying_a_run_that_does_not_exist_is_a_404(fresh_database, exported):
    client = client_for(fresh_database)
    # A worked run only to mint a valid token; we post to a different, unknown id.
    other = worked_run(fresh_database)
    token = token_on(client, f"/runs/{other}")
    unknown = "00000000-0000-0000-0000-000000000000"

    response = client.post(f"/runs/{unknown}/replay", data={"csrf": token}, follow_redirects=False)

    assert response.status_code == 404
