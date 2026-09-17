# The application image. The screen and the worker both run from it; only the command differs.
#
# psycopg ships as psycopg-binary here, which bundles libpq, so the image needs no Postgres client
# libraries of its own. Ollama is a separate service reached over the network (OPSAGENT_OLLAMA_URL),
# so no model or GPU tooling lives in here either -- this stays a small, plain Python image.

FROM python:3.13-slim

WORKDIR /app

# Dependencies first and on their own layer, so editing code does not reinstall them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only what production runs: the app, the migrations it applies, and the text it reads at load --
# the prompts (hashed at startup) and the policy documents (ingested into the store).
COPY app ./app
COPY migrations ./migrations
COPY prompts ./prompts
COPY policies ./policies

# Never as root. A fixed high uid so a bind-mounted volume's files have a predictable owner.
RUN useradd --create-home --uid 10001 opsagent && chown -R opsagent:opsagent /app
USER opsagent

# Python should not buffer stdout/stderr in a container, or logs arrive only when it exits.
ENV PYTHONUNBUFFERED=1

# The screen by default; the worker service overrides the command in the compose file.
EXPOSE 8055
CMD ["python", "-m", "app.web"]
