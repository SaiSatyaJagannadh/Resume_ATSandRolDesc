"""Single place the LLM provider is chosen. Switching providers is one env var."""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

import config


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return the chat model for the configured provider.

    Fails fast with a clear message if the provider's key is missing, rather
    than surfacing an opaque auth error deep inside a graph node.
    """
    provider = config.LLM_PROVIDER
    config.require_key(provider)
    model = config.MODELS[provider]

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Claude 4.6+ models reject the temperature parameter outright, so it is
        # simply not passed. Structured output is what constrains determinism
        # here anyway.
        return ChatAnthropic(model=model, max_tokens=16000)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    # Unreachable: require_key already validated the provider name.
    raise ValueError(f"Unsupported provider: {provider}")


def structured(schema, temperature: float = 0.0):
    """LLM constrained to return `schema` (a Pydantic model), not free text."""
    return get_llm(temperature).with_structured_output(schema)
