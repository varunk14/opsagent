"""
Would this reply have anywhere to go?

Sending SMTP to an address that does not exist costs three things. A bounce comes back and,
with the mailbox wired both directions, becomes a fresh run for the agent to answer. The
bouncing server's opinion of the sending IP gets a little worse. And the drain retries the
send four more times before giving up, because the underlying failure looks transient.

So each reply is checked before the SMTP session opens. Two layers, cheapest first:

  - reserved test domains (example.com, .invalid, .test, .localhost, .example) are refused by
    string match. The standard says nothing routes there; the fixture messages that seeded
    this project use `@example.com`, and the check catches them before any DNS lookup.

  - real domains are asked for their MX record. A domain that publishes no way to receive
    mail is one whose replies would only bounce. The resolver is passed in so the module
    itself never dials DNS -- tests hand it a stub, production hands it a dnspython wrapper.

`Undeliverable` is raised at the edge (the sender), not here. This module answers yes or no.
"""

from typing import Protocol

# RFC 2606 reserves these top-level and second-level names for testing and documentation. Nothing
# on the public internet routes there, so any reply addressed to one of them would be lost.
_RESERVED_TLDS = frozenset({"test", "example", "invalid", "localhost"})
_RESERVED_DOMAINS = frozenset({"example.com", "example.org", "example.net"})


class Resolver(Protocol):
    """The DNS reach this module needs. Anything that can answer `resolve_mx` fits."""

    def resolve_mx(self, domain: str) -> list[str]:
        """The MX hostnames for `domain`, or [] if it has none. Raises LookupError for NXDOMAIN."""


class Undeliverable(Exception):
    """The recipient address has nowhere for mail to land. Raised at the sender, caught by the drain."""


def _split(address: str) -> tuple[str, str] | None:
    """(local, domain) if the address has both halves, else None. No RFC 5321 gymnastics."""
    if not address or address.count("@") != 1:
        return None
    local, domain = address.split("@", 1)
    if not local or not domain:
        return None
    return local, domain.lower()


def _reserved_reason(domain: str) -> str | None:
    """The reserved-name reason for a lowercased domain, or None. Shared by both entry points."""
    for reserved in _RESERVED_DOMAINS:
        if domain == reserved or domain.endswith("." + reserved):
            return f"reserved test domain: {reserved}"
    tld = domain.rsplit(".", 1)[-1]
    if tld in _RESERVED_TLDS:
        return f"reserved test TLD: .{tld}"
    return None


def is_reserved(address: str) -> str | None:
    """A reason string if the address's domain is a reserved test name; None if it isn't."""
    parts = _split(address)
    if parts is None:
        return None
    return _reserved_reason(parts[1])


def check_deliverable(address: str, resolver: Resolver | None) -> str | None:
    """
    A reason the address cannot be delivered to, or None to mean "go ahead".

    The check is layered so that a reserved address never hits DNS -- the resolver may be None,
    which is how the sender says "I have no resolver wired, only run the free checks". In practice
    production always passes a resolver; the shape is here so tests do not need one either.
    """
    parts = _split(address)
    if parts is None:
        return f"not a well-formed address: {address!r}"

    domain = parts[1]
    reserved = _reserved_reason(domain)
    if reserved is not None:
        return reserved

    if resolver is None:
        return None

    try:
        hosts = resolver.resolve_mx(domain)
    except LookupError:
        return f"no such domain: {domain}"
    if not hosts:
        return f"no MX record for {domain}"
    return None
