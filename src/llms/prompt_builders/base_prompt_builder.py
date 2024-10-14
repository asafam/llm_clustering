from typing import *
from llms.utils import *

class BasePromptBuilder:
    def __init__(self, prompt_type) -> None:
        self.prompt_type = prompt_type
        # Load the prompts
        self.template_prompts = load_prompts(prompt_type=prompt_type)
        
    def build_prompt(self, step: int = 0, **kwargs):       
        # Select the prompt to build
        prompt = self.template_prompts[step]

        # Format hint
        hint = kwargs.get('hint', "their meaning")
        prompt = prompt.replace("{hint}", hint) # Replace the {hint} placeholder
        
        # Format texts as [ID: {index}] {text}
        ids = kwargs.get('ids')
        texts = kwargs.get('texts')
        if ids and texts:
            formatted_texts = "\n".join([f"[ID: {id}] {text}" for id, text in zip(ids, texts)])
            prompt = prompt.replace("{texts}", formatted_texts) # Replace the {texts} placeholder

        # Format clusters
        clusters = kwargs.get("clusters")
        if clusters:
            formatted_text = "{"
            formatted_text += ',\n'.join([('\t' + str(label) + ': [' + ', '.join([str(sid) for sid in clusters[label]]) + ']') for label in clusters.keys()])
            formatted_text += "}"
            prompt = prompt.replace("{clusters}", formatted_text)

        # Format k_info 
        k = kwargs.get('k')
        prompt = prompt.replace("{k_info}\n\n", f"Number of clusters: {k}\n\n" if (k is not None and k > 0) else "") # Replace the {k_info} placeholder

        return prompt
    
    def get_priority(self) -> int:
        return 0
    
    def get_steps_count(self) -> int:
        return len(self.template_prompts)
    
    def is_match(self, prompt_type: PromptType) -> bool:
        return True