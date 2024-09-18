from typing import *
from llms.models import *
from llms.utils import *
from experiments.utils import get_prompt_type
from clustering.constraints import *
from clustering.constraints_manager import ConstraintsType, generate_constraint


class BaseConstrainedLLM:
    def __init__(self, llm: LLM, constraint_type: ConstraintsType) -> None:
        self.llm = llm
        self.constraint_type = constraint_type
    

    def create_constraint(self, labels: Optional[list] = None, prompt_type: Optional[PromptType] = None, **kwargs):
        # generate the prompt
        prompt_type = prompt_type or get_prompt_type(constraint_type=self.constraint_type)
        prompt = generate_prompt(prompt_type=prompt_type, **kwargs)

        # execute prompt (possibly n times)
        data0 = self.llm.create_messages(prompt=prompt)

        formatter = get_formatter(prompt_type=prompt_type)
        data = formatter(data0)

        # process results and generate constraints
        constraint = generate_constraint(data=data, constraint_type=self.constraint_type, **kwargs)
        constraint_quality = constraint.evaluate(true_labels=labels) if labels else {}
        result = dict(
            constraint=constraint,
            prompt=prompt,
            constraint_quality = constraint_quality
        )
        return result

        

        

