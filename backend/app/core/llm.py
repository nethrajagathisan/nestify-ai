from functools import lru_cache
from typing import Optional
from groq import Groq, APIError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import get_settings


class GroqClient:
    def __init__(self, api_key: str, chat_model: str, classifier_model: str):
        self._client: Optional[Groq] = None
        self._api_key = api_key
        self._chat_model = chat_model
        self._classifier_model = classifier_model

    def _get_client(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=self._api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, APIConnectionError)),
    )
    def chat(
        self,
        messages: list[dict],
        system_prompt: str,
        model: Optional[str] = None,
    ) -> str:
        client = self._get_client()
        
        # Prepend system prompt
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Use provided model or default chat model
        model_to_use = model or self._chat_model
        
        response = client.chat.completions.create(
            model=model_to_use,
            messages=full_messages,
        )
        
        return response.choices[0].message.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, APIConnectionError)),
    )
    def classify(self, prompt: str) -> str:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self._classifier_model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        
        return response.choices[0].message.content


@lru_cache
def get_groq_client() -> GroqClient:
    settings = get_settings()
    return GroqClient(
        api_key=settings.GROQ_API_KEY,
        chat_model=settings.GROQ_CHAT_MODEL,
        classifier_model=settings.GROQ_CLASSIFIER_MODEL,
    )
