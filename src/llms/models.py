from anthropic import AnthropicBedrock
import ast
import logging


class LLM:
    def create_messages(self, prompt: str, **kwargs):
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return self.__class__.__name__
        

class Claude(LLM):
    def __init__(self, model, aws_region) -> None:
        super().__init__()
        self.model = model
        self.aws_region = aws_region

    def create_messages(self, prompt: str, max_tokens: int = 4096, temperature=0):
        logger = logging.getLogger('default')
        client = AnthropicBedrock(
            aws_region=self.aws_region,
        )

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        data_text = response.content[0].text
        logger.debug("LLM returned message")
        
        try:
            data = ast.literal_eval(data_text)
            return data
        except:
            raise Exception(f"Error casting the response to an object:\n{data_text}")
        



