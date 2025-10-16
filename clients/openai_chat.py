import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIChat:
    def __init__(self) -> None:
        self.model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, instructions: str, messages: list[dict]) -> str:
        """Send a query using Chat Completions and return assistant content."""
        try:
            completion = self.client.chat.completions.create(
                temperature=0.0,
                model=self.model,
                messages=[
                    {"role": "system", "content": instructions},
                    *[
                        {"role": message["role"], "content": message["content"]}
                        for message in messages
                    ],
                ],
            )
            if (
                completion.choices
                and completion.choices[0].message
                and completion.choices[0].message.content
            ):
                content = completion.choices[0].message.content
                return content or ""
        except Exception:
            logger.info("Error sending query to OpenAI. Returning empty string.")
            return ""
        return ""
