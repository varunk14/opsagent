"""
Refusing to deliver a reply to an address the DNS says is not real.

Two layers, in order of cost. Reserved test domains (example.com, .invalid, .test) are checked
by string match: nothing routes there and the standard says nothing will. Real domains are then
asked for their MX record. A domain that publishes no way to receive mail is one whose replies
would only bounce, and a bounce is worth avoiding: with SMTP wired both directions, the
bounce lands back in the mailbox and becomes a fresh run.

The resolver is passed in so this file never dials DNS. `dnspython` is what the real caller
uses; the fakes here return what the tests need.
"""

import pytest

from app.replies.deliverability import (
    Undeliverable,
    check_deliverable,
    is_reserved,
)


class StubResolver:
    """A DNS resolver that answers what the test rehearsed, and nothing else."""

    def __init__(self, mx: dict[str, list[str]] | None = None, nx: set[str] | None = None):
        self._mx = mx or {}
        self._nx = nx or set()

    def resolve_mx(self, domain: str) -> list[str]:
        if domain in self._nx:
            raise LookupError(f"no such domain: {domain}")
        return list(self._mx.get(domain, []))


# --- reserved domains -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "priya@example.com",
        "someone@example.org",
        "dev@example.net",
        "user@sub.example.com",
        "test@foo.test",
        "user@thing.invalid",
        "root@localhost",
        "person@somewhere.example",
    ],
)
def test_reserved_addresses_are_refused_without_dns(address):
    assert is_reserved(address) is not None


@pytest.mark.parametrize(
    "address",
    [
        "customer@gmail.com",
        "ops@company.co.uk",
        "hello@duckduckgo.com",
    ],
)
def test_real_addresses_are_not_reserved(address):
    assert is_reserved(address) is None


# --- MX check ---------------------------------------------------------------------------------


def test_check_deliverable_refuses_reserved_before_dns():
    assert check_deliverable("priya@example.com", resolver=None) is not None


def test_check_deliverable_accepts_domain_with_mx():
    resolver = StubResolver(mx={"gmail.com": ["gmail-smtp-in.l.google.com"]})
    assert check_deliverable("customer@gmail.com", resolver=resolver) is None


def test_check_deliverable_refuses_domain_without_mx():
    resolver = StubResolver(mx={"broken.example.co": []})
    reason = check_deliverable("someone@broken.example.co", resolver=resolver)
    assert reason is not None
    assert "MX" in reason or "mx" in reason


def test_check_deliverable_refuses_nonexistent_domain():
    resolver = StubResolver(nx={"nowhere.example.co"})
    reason = check_deliverable("someone@nowhere.example.co", resolver=resolver)
    assert reason is not None


def test_check_deliverable_refuses_malformed_address():
    resolver = StubResolver()
    assert check_deliverable("not-an-address", resolver=resolver) is not None
    assert check_deliverable("", resolver=resolver) is not None
    assert check_deliverable("@nodomain", resolver=resolver) is not None
    assert check_deliverable("nolocal@", resolver=resolver) is not None


def test_undeliverable_is_an_exception():
    with pytest.raises(Undeliverable):
        raise Undeliverable("no MX record for broken.example.co")
