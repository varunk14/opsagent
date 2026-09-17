"""
Where the model server lives, so the app can be deployed apart from it.

On this machine Ollama is on localhost, and that is the default. In a container it is a service of
its own reachable by name, so the base URL comes from OPSAGENT_OLLAMA_URL when it is set. Everything
else -- which model, which path -- stays fixed; only where the server is moves.
"""

from app.embeddings import OLLAMA_EMBED
from app.llm import OLLAMA, ollama_endpoint


def test_it_defaults_to_localhost():
    assert ollama_endpoint("api/generate", {}) == "http://localhost:11434/api/generate"


def test_the_base_comes_from_the_environment():
    endpoint = ollama_endpoint("api/generate", {"OPSAGENT_OLLAMA_URL": "http://ollama:11434"})
    assert endpoint == "http://ollama:11434/api/generate"


def test_an_empty_value_falls_back_like_an_unset_one():
    """A compose file with OPSAGENT_OLLAMA_URL= (blank) must not produce a scheme-relative URL."""
    assert ollama_endpoint("api/generate", {"OPSAGENT_OLLAMA_URL": ""}) == (
        "http://localhost:11434/api/generate"
    )


def test_a_trailing_slash_does_not_double_up():
    endpoint = ollama_endpoint("api/embed", {"OPSAGENT_OLLAMA_URL": "http://ollama:11434/"})
    assert endpoint == "http://ollama:11434/api/embed"


def test_the_module_endpoints_are_built_through_it():
    assert OLLAMA.endswith("/api/generate")
    assert OLLAMA_EMBED.endswith("/api/embed")
