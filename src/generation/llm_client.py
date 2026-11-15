import requests

from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt


class LLMClient:
    def __init__(self, model: str):
        self.model = model

    @classmethod
    def from_ollama(cls, model: str) -> "LLMClient":
        return cls(model)

    def complete(self, question: str, context: str) -> str:
        response = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, context)},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()
        completion = response.json()
        content = completion.get("message", {}).get("content")
        if completion.get("done") and completion.get("done_reason") == "stop" and content:
            return content
        raise ValueError("The model returned no complete answer.")
