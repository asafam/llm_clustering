from anthropic import AnthropicBedrock
import ast
import logging


class LLM:
    def create_messages(self, messages: list, **kwargs):
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return self.__class__.__name__
        

class Claude(LLM):
    def __init__(self, model, aws_region) -> None:
        super().__init__()
        self.model = model
        self.aws_region = aws_region
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, aws_region={self.aws_region})"

    def create_messages(self, messages: list, max_tokens: int = 4096, temperature=0):
        logger = logging.getLogger('default')
        client = AnthropicBedrock(
            aws_region=self.aws_region,
        )

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages
        )

        data_text = response.content[0].text
        logger.debug(f"LLM returned message (usage statistics: {response.usage})")
        
        try:
            data = ast.literal_eval(data_text)
            return data, data_text
        except:
            raise Exception(f"Error casting the response to an object:\n{data_text}")
        