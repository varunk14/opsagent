"""
The local screen: approve or reject what the guardrail would not, and read what runs did.

Local, and deliberately thin. It lists what app.approvals returns and records a
decision with approvals.decide. It never loads the executor or the worker: an
approved refund is paid by the next worker to claim the run, through the same
keyed executor and ledger cap as everything else, never by this process.

/runs lists the newest runs, and /runs/{id} shows one run's trace -- every tick,
step and model call with its cost -- read from the spans table through
app.traces, never from Langfuse. When OPSAGENT_LANGFUSE_PROJECT_URL names a
Langfuse project on this machine, a run's page links to the same trace there.

Everything shown came from a customer's email or a model's reading of one, so
Jinja escapes all of it and no template turns that off. A decision is a POST
carrying this process's token in both the form and a SameSite=Strict cookie, so a
page on another site can neither read the token nor send the cookie. A request
naming any host but this machine is refused, which stops DNS rebinding, and every
page forbids scripts, framing and posting anywhere else.

The person deciding is named by OPSAGENT_OPERATOR. Without it the screen still
shows what is waiting, but refuses to decide.

Run:  OPSAGENT_OPERATOR=asha .venv/bin/python -m app.web
      then open http://127.0.0.1:8055/approvals or http://127.0.0.1:8055/runs
"""

import hmac
import os
import secrets
import sys
from collections.abc import Awaitable, Callable
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit
from uuid import UUID

import jinja2
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.approvals import PendingApproval, decide, list_handed_over, list_pending
from app.db import connect
from app.failures import FIXES, GOLDEN_HISTORY, failure_chart, golden_trend, mix_by_week
from app.guardrails import Budgets, load, rupees
from app.traces import TraceSpan, list_runs, trace_of
from app.tracing import LOOPBACK_HOSTS, Attr

HOST = "127.0.0.1"
PORT = 8055
ALLOWED_HOSTS = [HOST, "localhost"]
CSRF_COOKIE = "opsagent_csrf"
LANGFUSE_VAR = "OPSAGENT_LANGFUSE_PROJECT_URL"

# Escaping is switched on for every template whatever its name, not left to the file extension.
# A detail that was never recorded reads as that, not as Python's "None".
TEMPLATES = Jinja2Templates(
    env=jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).resolve().parent / "templates"),
        autoescape=True,
        finalize=lambda value: "not recorded" if value is None else value,
    )
)


def money(amount: Decimal) -> str:
    """A reference cost exactly as recorded -- $0.0000045, not a rounded $0.00."""
    return "$" + format(amount.normalize(), "f")


TEMPLATES.env.filters["money"] = money
TEMPLATES.env.filters["rupees"] = rupees

# Shown for any unhandled error. Fixed text: nothing about the cause reaches the page.
FAILED_PAGE = (
    "<!doctype html><title>Something went wrong</title>"
    "<p>Something went wrong and nothing on this page was completed. "
    "The screen's own output says why. Reload to see the current state.</p>"
)

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    # Not "no-referrer": under it a browser sends "Origin: null" on this screen's own form posts,
    # and the Origin check below would refuse every decision. Other sites still get no referrer.
    "Referrer-Policy": "same-origin",
    # Customer emails are on these pages; they should not outlive the tab in a browser cache.
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
}

DECISIONS = {"approve": True, "reject": False}

# What a span did, in the order a person reads it, each with the label shown before it.
# Only values the agent chose from its own lists -- never text a customer wrote.
SHOWN = (
    (Attr.OUTCOME, "ended"),
    (Attr.ATTEMPT, "attempt"),
    (Attr.INTENT, "intent"),
    (Attr.CLASSIFICATION_CONFIDENCE, "confidence"),
    (Attr.ORDER_ID_FOUND, "order id found"),
    (Attr.AMOUNT_FOUND, "amount found"),
    (Attr.PASSAGES, "passages"),
    (Attr.TOOL, "tool"),
    (Attr.PROPOSAL_CONFIDENCE, "confidence"),
    (Attr.RESULT, "result"),
    (Attr.APPROVAL_ID, "approval"),
    (Attr.VERDICT, "guardrail"),
    (Attr.REASON, "why"),
    (Attr.FAILURE, "stopped"),
)


def approval_view(pending: PendingApproval) -> dict[str, Any]:
    """What one pending approval looks like to the person deciding. Plain values; the template escapes them."""
    args = pending.action.get("args", {})
    evidence = pending.evidence
    lookups = [step.get("result", {}) for step in evidence.get("steps", []) if step.get("tool") == "get_order"]
    ledger = lookups[-1] if lookups else {}
    amount = args.get("amount_paise")
    refunded = ledger.get("refunded_paise")
    return {
        "id": pending.id,
        "amount": rupees(amount) if isinstance(amount, int) else "an unknown amount",
        "order_id": args.get("order_id", "unknown"),
        "refund_reason": args.get("reason"),
        "charges": [rupees(charge) for charge in ledger.get("charges_paise", []) if isinstance(charge, int)],
        "refunded": rupees(refunded) if isinstance(refunded, int) else None,
        "confidence": str(pending.confidence) if pending.confidence is not None else "none given",
        "reason": pending.reason,
        "sender": evidence.get("sender"),
        "subject": evidence.get("subject"),
        "body": evidence.get("body"),
        "asked_at": pending.created_at,
    }


def span_view(node: TraceSpan, depth: int) -> dict[str, Any]:
    """One trace row as the run page shows it. Plain values; the template escapes them."""
    if node.kind == "generation" and node.own_cost is not None:  # the table refuses an uncosted generation
        cost = money(node.own_cost)
    elif node.kind == "embedding":
        cost = "not counted"  # embedding tokens are not in the run's cost yet; blank would read as free
    else:
        cost = ""
    counted = node.input_tokens is not None and node.output_tokens is not None
    did = [f"{label}: {node.attributes[key]}" for key, label in SHOWN if key in node.attributes]
    if node.status == "error":
        did.insert(0, f"failed: {node.status_message or 'no reason recorded'}")
    return {
        "depth": depth,
        "name": node.name,
        "error": node.status == "error",
        "duration_ms": node.duration_ms,
        "model": node.model or "",
        "prompt_version": node.prompt_version or "",
        "tokens": f"{node.input_tokens} in, {node.output_tokens} out" if counted else "",
        "cost": cost,
        "did": did,
    }


def trace_rows(roots: list[TraceSpan]) -> list[dict[str, Any]]:
    """The trace as table rows, each parent before its children, each at its depth."""
    rows: list[dict[str, Any]] = []
    stack = [(0, node) for node in reversed(roots)]
    while stack:
        depth, node = stack.pop()
        rows.append(span_view(node, depth))
        stack.extend((depth + 1, child) for child in reversed(node.children))
    return rows


def langfuse_link_base(url: str | None) -> str | None:
    """Where a run's trace can also be opened in Langfuse, or None. Only a Langfuse on this machine is linked."""
    if url is None or not url.strip():
        return None
    parts = urlsplit(url.strip())
    if parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS:
        raise ValueError(
            "a Langfuse link must point at this machine (http://127.0.0.1 or http://localhost), "
            f"not {parts.scheme}://{parts.hostname}"
        )
    return url.strip().rstrip("/")


NO_CEILING = "no ceiling"


def spending(budgets: Budgets) -> list[tuple[str, str]]:
    """
    What one run may spend, ready to read. Zero is said in words, never shown as a number.

    Zero means no ceiling here, the opposite of the refund limit where zero stops everything, and
    "0 tokens" on the page would read as the harshest setting there is rather than the absence of
    one.
    """
    return [
        ("Tokens one run may spend", f"{budgets.max_tokens_per_run:,}" if budgets.max_tokens_per_run else NO_CEILING),
        (
            "Reference cost one run may reach",
            f"${budgets.max_cost_usd_per_run:f}" if budgets.max_cost_usd_per_run else NO_CEILING,
        ),
        (
            "Seconds of model time one run may take",
            f"{budgets.max_seconds_per_run} seconds" if budgets.max_seconds_per_run else NO_CEILING,
        ),
    ]


def create_app(
    dsn: str | None = None,
    operator: str | None = None,
    langfuse_project_url: str | None = None,
    history_path: Path | None = None,
) -> FastAPI:
    """The screen, with a token of its own. `dsn` defaults to OPSAGENT_DATABASE_URL."""
    history = history_path or GOLDEN_HISTORY
    langfuse = langfuse_link_base(langfuse_project_url)
    token = secrets.token_urlsafe(32)
    name = (operator or "").strip() or None
    app = FastAPI(title="OpsAgent", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.exception_handler(Exception)
    async def failed(request: Request, error: Exception) -> Response:
        # Starlette answers an unhandled error outside every middleware, so the headers are attached here as well.
        return HTMLResponse(FAILED_PAGE, status_code=500, headers=SECURITY_HEADERS)

    def page(request: Request, template: str, context: dict[str, Any], status_code: int = 200) -> Response:
        response = TEMPLATES.TemplateResponse(
            request, template, {**context, "csrf": token, "operator": name}, status_code=status_code
        )
        response.set_cookie(CSRF_COOKIE, token, httponly=True, samesite="strict", path="/")
        return response

    def message(request: Request, text: str, status_code: int) -> Response:
        return page(request, "message.html", {"text": text}, status_code)

    @app.get("/approvals", response_class=HTMLResponse)
    def approvals(request: Request) -> Response:
        with connect(dsn) as connection:
            pending = [approval_view(item) for item in list_pending(connection)]
            handed_over = list_handed_over(connection)
        return page(request, "approvals.html", {"pending": pending, "handed_over": handed_over})

    @app.post("/approvals/{approval_id}/{choice}")
    def record(
        request: Request,
        approval_id: int,
        choice: str,
        csrf: Annotated[str, Form()] = "",
        note: Annotated[str, Form()] = "",
    ) -> Response:
        origin = request.headers.get("origin")
        if origin is not None and origin != str(request.base_url).rstrip("/"):
            return message(request, "This form was sent from another site. Nothing was recorded.", 403)
        cookie = request.cookies.get(CSRF_COOKIE, "")
        if not (hmac.compare_digest(csrf.encode(), token.encode()) and hmac.compare_digest(cookie.encode(), token.encode())):
            return message(request, "This form did not come from this screen, or the screen has restarted. Reload the page.", 403)
        if choice not in DECISIONS:
            return message(request, "There is no such decision.", 404)
        if name is None:
            return message(request, "Set OPSAGENT_OPERATOR to your name and restart the screen, so a decision records who made it.", 403)
        try:
            with connect(dsn) as connection:
                decided = decide(connection, approval_id, approved=DECISIONS[choice], by=name, note=note.strip() or None)
        except ValueError as refused:
            return message(request, f"Not recorded: {refused}.", 400)
        if not decided:
            return message(request, "Not recorded: this approval was already decided, or its run has moved on.", 409)
        return RedirectResponse("/approvals", status_code=303)

    @app.get("/guardrails", response_class=HTMLResponse)
    def guardrails(request: Request) -> Response:
        with connect(dsn) as connection:
            limits = load(connection)
        return page(
            request,
            "guardrails.html",
            {
                "limit": rupees(limits.auto_refund_limit_paise),
                "min_confidence": str(limits.min_confidence),
                "budgets": spending(limits.budgets),
            },
        )

    @app.get("/failures", response_class=HTMLResponse)
    def failures(request: Request) -> Response:
        with connect(dsn) as connection:
            rows = mix_by_week(connection)
        return page(
            request,
            "failures.html",
            {
                "weeks": failure_chart(rows),
                "total": sum(count for _, _, count in rows),
                "trend": golden_trend(history),
                "fixes": FIXES,
            },
        )

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request) -> Response:
        with connect(dsn) as connection:
            recent = list_runs(connection)
        return page(request, "runs.html", {"runs": recent})

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run(request: Request, run_id: str) -> Response:
        # Parsed before any connection is opened: an id that is not a run's never reaches the database.
        try:
            parsed = UUID(run_id)
        except ValueError:
            return message(request, "No such run.", 404)
        with connect(dsn) as connection:
            trace = trace_of(connection, parsed)
        if trace is None:
            return message(request, "No such run.", 404)
        return page(
            request,
            "run.html",
            {
                "trace": trace,
                "rows": trace_rows(trace.roots),
                "langfuse_url": f"{langfuse}/traces/{parsed.hex}" if langfuse else None,
            },
        )

    return app


def main() -> int:  # pragma: no cover - serves until stopped
    operator = os.environ.get("OPSAGENT_OPERATOR", "").strip() or None
    if operator is None:
        print("  OPSAGENT_OPERATOR is not set: the screen shows what is waiting but will not record decisions")
    print(f"  approvals on http://{HOST}:{PORT}/approvals, runs on http://{HOST}:{PORT}/runs")
    app = create_app(operator=operator, langfuse_project_url=os.environ.get(LANGFUSE_VAR))
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
