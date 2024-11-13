from typing import *
from llms.utils import *

class BasePromptBuilder:
    def __init__(self, prompt_type) -> None:
        self.prompt_type = prompt_type
        # Load the prompts
        self.steps_prompts = load_prompts(prompt_type=prompt_type)

    def __str__(self) -> str:
        return self.__class__.__name__
    
    def get_prompts(self, step: int = 0):
        # Translate the step to a prompt index as the step or the last step if we reached the end
        step_index = step if step < len(self.steps_prompts) else -1

        step_prompts = next((step_prompts for step_prompts in self.steps_prompts if step_prompts.get('step') == step_index), None)
        if step_prompts is None:
            # Get the step prompts by the index
            step_prompts = self.steps_prompts[step_index]
            
        step_prompts = step_prompts.copy()
        return step_prompts
        
    def build_prompt(self, system_prompt: str, user_prompt: str, step: int = 0, **kwargs):       
        prompt_params = dict()

        prompt_params["hint"] = kwargs.get('hint', "their meaning")
        prompt_params["dataset_knowledge"] = kwargs.get('dataset_knowledge')
        
        # Format texts as [ID: {index}] {text}
        ids = kwargs.get('ids')
        texts = kwargs.get('texts')
        if ids and texts:
            formatted_texts = "\n".join([f"[ID: {id}] {text}" for id, text in zip(ids, texts)])
            prompt_params["texts"] = formatted_texts

        # Format clusters
        clusters = kwargs.get("clusters")
        if clusters:
            formatted_text = "{"
            formatted_text += ',\n'.join([('\t' + str(label) + ': [' + ', '.join([str(x['id']) for x in clusters[label]]) + ']') for label in clusters.keys()])
            formatted_text += "}"
            prompt_params["clusters_by_ids"] = formatted_text

            formatted_text = "{"
            formatted_text += ',\n'.join([('\t' + str(label) + ': [' + '\n\t'.join([f"[ID: {x['id']}] {x['text']}" for x in clusters[label]]) + '\n]') for label in clusters.keys()])
            formatted_text += "}"
            prompt_params["clusters_by_texts"] = formatted_text

        # Format explanations
        explanations = kwargs.get("explanations")
        if explanations:
            formatted_text = ""
            formatted_text += ',\n'.join([f" - {str(label)}: {e}" for label, e in explanations.items()])
            prompt_params["explanations"] = formatted_text

        # Format k_info 
        k = kwargs.get('k')
        prompt_params["k_info"] = f"Number of clusters: {k}\n\n" if (k is not None and k > 0) else "" # Replace the {k_info} placeholder

        system_prompt = system_prompt.format(**prompt_params)
        user_prompt = user_prompt.format(**prompt_params)

        return system_prompt, user_prompt
    
    def get_priority(self) -> int:
        return 0
    
    def get_steps_count(self) -> int:
        return len(self.template_prompts)
    
    def is_match(self, prompt_type: PromptType) -> bool:
        return True