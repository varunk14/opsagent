#!/usr/bin/env bash
# Bring the whole thing up on a fresh host, from nothing to a working screen.
#
# This is the manual deploy: you run it, on the box, over SSH. There is deliberately no key in
# GitHub and no pipeline that can reach production -- the thing that deploys is a person who is
# already on the machine. Re-run it after a change; it rebuilds the image and re-applies only what
# moved, and the migrations and model pulls are safe to run again.
set -euo pipefail

cd "$(dirname "$0")/.."   # the repo root, wherever this checkout lives

COMPOSE="docker compose -f compose.deploy.yml"

if [ ! -f .env ]; then
	echo "No .env found in the repo root." >&2
	echo "Copy deploy/.env.deploy.example to .env and fill it in first." >&2
	exit 1
fi

echo "==> Building the app image and starting the stack"
$COMPOSE up -d --build

echo "==> Waiting for the model server, then pulling the models (the first run downloads a few GB)"
$COMPOSE exec -T ollama sh -c 'until ollama list >/dev/null 2>&1; do sleep 2; done'
$COMPOSE exec -T ollama ollama pull llama3.1:8b
$COMPOSE exec -T ollama ollama pull nomic-embed-text

echo "==> Applying migrations and loading the policy documents"
$COMPOSE exec -T web python -m app.policies

echo "==> Seeding the demo ledger"
$COMPOSE exec -T web python -m app.seed

echo
echo "Up. Point the domain's DNS at this host if you have not -- Caddy needs it to get a"
echo "certificate -- then open the domain and sign in with the password you set in .env."
