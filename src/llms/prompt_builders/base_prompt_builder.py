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

        examples = kwargs.get("prompt_examples")
        example = examples[step if step == 0 else 1]
        prompt_params["example"] = self.build_example(example)
        
        # Format texts as [ID: {index}] {text}
        ids = kwargs.get('ids')
        texts = kwargs.get('texts')
        if ids and texts:
            formatted_texts = "\n".join([f"[ID: {id}] {text}" for id, text in zip(ids, texts)])
            prompt_params["texts"] = formatted_texts
            
        # Format clusters
        clusters = kwargs.get("clusters")
        if clusters:
            formatted_text = "{{"
            formatted_text += ',\n'.join([('\t' + str(label) + ': [' + ', '.join([str(x['id']) for x in clusters[label]]) + ']') for label in clusters.keys()])
            formatted_text += "}}"
            prompt_params["clusters_by_ids"] = formatted_text

            formatted_text = "{{"
            formatted_text += ',\n'.join([('\t' + str(label) + ': [' + '\n\t'.join([f"[ID: {x['id']}] {x['text']}" for x in clusters[label]]) + '\n]') for label in clusters.keys()])
            formatted_text += "}}"
            prompt_params["clusters_by_texts"] = formatted_text
        else:
            prompt_params["clusters_by_ids"] = "{{ }}"
            prompt_params["clusters_by_texts"] = "{{ }}"

        # Format explanations
        explanations = kwargs.get("explanations")
        if explanations:
            formatted_text = ""
            formatted_text += ',\n'.join([f" - {str(label)}: {e}" for label, e in explanations.items()])
            prompt_params["explanations"] = formatted_text

        # Format k_info 
        n_clusters = kwargs.get('n_clusters')
        prompt_params["k_info"] = f"the goal is to partition the data into exactly {n_clusters} clusters. However, the provided sample data may contain fewer than {n_clusters} clusters due to its limited size.\n\n" if (n_clusters is not None and n_clusters > 0) else "" # Replace the {k_info} placeholder

        system_prompt = system_prompt.format(**prompt_params)
        user_prompt = user_prompt.format(**prompt_params)

        return system_prompt, user_prompt
    
    def build_example(self, example_obj) -> str:
        existing_clusters = example_obj.get('existing_clusters')
        sentences = example_obj['sentences']
        explanations = example_obj['explanations']

        example = "<example>\n"
        
        if existing_clusters:
            example += "Existing clustering:\n"
            example += "{{\n"
            for label, ids in existing_clusters.items():
                example += f"\t{str(label)}: [{', '.join([str(x) for x in ids])}]\n"
            example += "}}\n"
            example += "\n"

        example += "Sentences:\n"
        for sentence in sentences:
            example += f"[ID: {sentence['id']}] {sentence['text']}\n"
        example += "\n"
        example += "Outputs:\n"
        example += "{{\n"
        example += "\t\"results\": {{\n"
        for sentence in sentences:
            example += f"\t\t{sentence['id']}: {sentence['label']},\n"
        example += "\t}}\n"
        example += "}}\n"
        example += "\n"

        example += "Explanations:\n"
        for explanation in explanations:
            example += f"- {explanation}\n"

        example += "</example>"
        
        return example

    def get_priority(self) -> int:
        return 0
    
    def get_steps_count(self) -> int:
        return len(self.template_prompts)
    
    def is_match(self, prompt_type: PromptType) -> bool:
        return True