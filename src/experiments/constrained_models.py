from typing import *
from llms.models import *
from llms.utils import *
from experiments.utils import get_prompt_type
from clustering.constraints import *
from clustering.constraints_manager import ConstraintsType, generate_constraint
import logging

class BaseConstrainedLLM:
    def __init__(self, llm: LLM, constraint_type: ConstraintsType) -> None:
        self.llm = llm
        self.constraint_type = constraint_type
    

    def create_constraint(self, ids: Optional[list] = None, labels: Optional[list] = None, prompt_type: Optional[PromptType] = None, **kwargs):
        logger = logging.getLogger('default')
        # generate the prompt
        prompt_type = prompt_type or get_prompt_type(constraint_type=self.constraint_type)
        prompt = generate_prompt(prompt_type=prompt_type, ids=ids, **kwargs)

        # execute prompt (possibly n times)
        data = self.llm.create_messages(prompt=prompt)

        format_func = get_formatter(prompt_type=prompt_type)
        data = format_func(data) if format_func else data

        # process results and generate constraints
        constraint = generate_constraint(data=data, constraint_type=self.constraint_type, **kwargs)
        constraint_quality = constraint.evaluate(ids_true=ids, labels_true=labels) if labels else {}
        logger.debug(f"Constraint evaluation returned {constraint_quality}")
        result = dict(
            constraint=constraint,
            prompt=prompt,
            constraint_quality = constraint_quality
        )
        return result

        

        

