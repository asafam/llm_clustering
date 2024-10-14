from typing import *
from llms.utils import PromptType
from llms.prompt_builders.base_prompt_builder import BasePromptBuilder
import logging

class MultistepPromptBuilder(BasePromptBuilder):
    def build_prompt(self, step: int = 0, context: any = None, **kwargs):
        if step == 0:
            prompt = super().build_prompt(step=step, **kwargs)
        elif step == 1:
            # prepare the 'cluters' arg
            sentences_labels = context.get("sentences_labels")
            clusters = DefaultDict(list)
            for sentence_id, label in sentences_labels.items():
                clusters[label].append(sentence_id)
                
            prompt = super().build_prompt(step=step, clusters=clusters, **kwargs)
        else:
            raise ValueError(f"")

        return prompt
    
    def get_priority(self) -> int:
        return (super().get_priority() + 1)
    
    def is_match(self, prompt_type: PromptType) -> bool:
        return prompt_type in [PromptType.MultiStepHardLabelsClusteringPrompt]

    