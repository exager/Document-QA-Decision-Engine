class BaseLLM:
    def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError