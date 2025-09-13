# Gemini 2.5 Flash 串接服務
from app.utils.llm_utils import safe_generate


class LLMService:
    def generate(self, prompt: str) -> str:
        result = safe_generate(prompt)
        return result or "AI 回應"
