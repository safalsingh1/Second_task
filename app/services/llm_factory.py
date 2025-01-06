from typing import Any, Dict, List, Type

import instructor
from pydantic import BaseModel

from config.settings import get_settings
from .ollama_chat import OllamaChat


class LLMFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        client_initializers = {
            "ollama": lambda s: OllamaChat(model_name=s.default_model),
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat_completion_create(**completion_params)
