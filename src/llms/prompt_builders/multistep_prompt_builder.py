from typing import *
from llms.utils import PromptType
from llms.prompt_builders.base_prompt_builder import BasePromptBuilder
import logging

class MultistepPromptBuilder(BasePromptBuilder):
    def build_prompt(self, system_prompt: str, user_prompt: str, step: int = 0, context: any = None, **kwargs):
        if step == 0:
            system_prompt, user_prompt = super().build_prompt(system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
        elif context is not None:
            # prepare the 'cluters' arg
            sentences_labels = context.get("sentences_labels")
            ids_texts = context.get("ids_texts")
            explanations = context.get("explanations")
            clusters = DefaultDict(list)
            for sentence_id, label in sentences_labels.items():
                clusters[label].append({'id': sentence_id, 'text': ids_texts[sentence_id]})
                
            system_prompt, user_prompt = super().build_prompt(system_prompt=system_prompt, user_prompt=user_prompt, clusters=clusters, explanations=explanations, **kwargs)
        else:
            raise ValueError(f"Context should have a value for the sentences_labels key, current value is {context}")

        return system_prompt, user_prompt
    
    def get_priority(self) -> int:
        return (super().get_priority() + 1)
    
    def is_match(self, prompt_type: PromptType) -> bool:
        return prompt_type in [PromptType.MultiStepHardLabelsClusteringPrompt, PromptType.MultiStepHardLabelsClusteringPrompt2, PromptType.MultiStepHardLabelsClusteringPrompt3]

    