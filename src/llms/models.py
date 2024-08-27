import anthropic
import os
import ast


class LLM:
    def create_messages(self, prompt: str, **kwargs):
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return self.__class__.__name__
        

class Claude(LLM):
    def create_messages(self, prompt: str, max_tokens: int = 1024, temperature=0):
        client = anthropic.Anthropic()

        response = client.messages.create(
            model=os.getenv("AWS_CLAUDE_MODEL"),
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
        
        try:
            data = ast.literal_eval(data_text)
            return data
        except:
            raise Exception(f"Error casting the response to an object:\n{data_text}")
        


