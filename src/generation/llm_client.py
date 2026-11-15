from openai import OpenAI

from src import config
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt


class LLMClient:
    def __init__(self, model: str, *, client: OpenAI | None = None):
        self.model = model
        self.client = client if client is not None else OpenAI(
            api_key=config.OPENAI_API_KEY, timeout=30.0
        )

    def complete(self, question: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, context)},
            ],
            response_format={"type": "json_object"},
        )
        if response.choices:
            choice = response.choices[0]
            if choice.finish_reason == "stop" and choice.message.content:
                return choice.message.content
        raise ValueError("The model returned no complete answer.")
