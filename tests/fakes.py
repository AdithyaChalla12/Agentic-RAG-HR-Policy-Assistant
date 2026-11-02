"""A scripted fake LLM: returns queued answers in order, so tests need no API keys."""
from langchain_core.messages import AIMessage


class FakeLLM:
    def __init__(self, structured: list, texts: list[str] | None = None):
        self.structured = list(structured)  # pydantic objects returned by with_structured_output
        self.texts = list(texts or ["fake answer"])
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        llm = self

        class _Runner:
            def invoke(self, prompt):
                llm.prompts.append(prompt)
                out = llm.structured.pop(0)
                assert isinstance(out, schema), f"expected {schema.__name__}, got {type(out).__name__}"
                return out

        return _Runner()

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.texts.pop(0) if self.texts else "fake answer")
