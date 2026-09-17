# Deploying opsagent

The whole system runs from one compose file behind Caddy, which terminates TLS and puts a password
in front of the screen. This is a **manual** deploy: you run one script, on the box, over SSH. There
is no key in GitHub and no pipeline that can reach production -- the thing that deploys is a person
already on the machine.

## What you need first

1. **A host with Docker.** An Oracle Cloud Always Free `A1.Flex` (ARM) instance is enough: a few
   cores and a few GB of RAM run Postgres, Redis, Ollama and the app together. Any Linux host with
   Docker and Docker Compose works.
2. **A domain pointed at it.** A free [DuckDNS](https://www.duckdns.org) name works
   (`opsagent-yourname.duckdns.org`). Point its DNS at the host's public IP **before** deploying --
   Caddy needs the name to resolve to get a TLS certificate.
3. **Ports 80 and 443 open** to the host (security list / firewall), for the certificate and the
   screen. Nothing else needs to be open.

## Steps

```bash
git clone https://github.com/varunk14/opsagent.git
cd opsagent
cp deploy/.env.deploy.example .env      # then edit .env; it is never committed

# Generate the screen password's hash and put it in .env as OPSAGENT_BASICAUTH.
# Note: compose reads a single '$' as a variable, so DOUBLE every '$' from the hash in .env.
docker run --rm caddy caddy hash-password --plaintext 'the-password-you-want'

./deploy/deploy.sh                      # build, start, pull the models, migrate, seed
```

`deploy.sh` is safe to re-run after a change: it rebuilds the image and re-applies only what moved.

## What is exposed

Only Caddy, on 80 and 443. The database, cache, model server, screen and worker all talk over the
internal compose network and are never reachable from outside. The screen answers only to the
domain (TrustedHost) and only after the password (Caddy basic auth), over TLS.

## Channels

Email and Telegram are optional. Fill their values in `.env` (see the README's "Real channels" for
the quoting rules) and the worker starts polling them; leave them blank and the demo runs on its
own fixtures. Refunds go out by email, enquiries are answered on Telegram.

## Updating

```bash
git pull
./deploy/deploy.sh
```
