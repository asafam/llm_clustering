from typing import *
from llms.models import *
from llms.utils import generate_prompt
from experiments.utils import get_prompt_type
from clustering.constraints import *
from clustering.constraints_manager import ConstraintsType, generate_constraint


class BaseConstrainedLLM:
    def __init__(self, llm: LLM, constraint_type: ConstraintsType) -> None:
        self.llm = llm
        self.constraint_type = constraint_type
    

    def create_constraint(self, texts: List[str], k: Optional[int]):
        # generate the prompt
        prompt_type = get_prompt_type(constraint_type=self.constraint_type)
        prompt = generate_prompt(prompt_type=prompt_type, texts=texts, k=k)

        # execute prompt (possibly n times)
        data = self.llm.create_messages(prompt=prompt)

        # process results and generate constraints
        constraint = generate_constraint(data, self.constraint_type)
        return constraint

        

        

