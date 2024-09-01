from enum import Enum
import os
import yaml
import logging


class PromptType(Enum):
    SimpleClusteringPrompt = 'simple_clustering_prompt1'
    HardLabelsClusteringPrompt = 'hard_labels_clustering_prompt'
    FuzzyLabelsClusteringPrompt = 'fuzzy_labels_clustering_prompt'
    MustLinkCannotLinkClusteringPrompt = 'must_link_cannot_link_clustering_prompt'


def generate_prompt(prompt_type: PromptType, text_index_offset: int = 1, **kwargs):
    logger = logging.getLogger('default')
    # Load the YAML file
    prompt_file = os.path.join(os.getenv('LLM_CLUSTERING_BASE_DIR', ''), 'prompts', f'{prompt_type.value}.yaml')
    logger.debug(f"Loading prompt file {prompt_file}")
    with open(prompt_file, 'r') as file:
        data = yaml.safe_load(file)
        template_prompt = data[0]['prompt']

    # Format the texts as [ID: {index}] {text}
    texts = kwargs.get('texts')
    text_index_offset = kwargs.get('text_index_offset', 1)
    formatted_texts = "\n".join([f"[ID: {index}] {text}" for index, text in enumerate(texts, start=text_index_offset)])
    prompt = template_prompt.replace("{texts}", formatted_texts) # Replace the {texts} placeholder

    # Format the k_info 
    k = kwargs.get('k')
    prompt = prompt.replace("{k_info}\n\n", f"Number of clusters: {k}\n\n" if (k is not None and k > 0) else "") # Replace the {k_info} placeholder

    return prompt


def format_response_as_dictionary_of_clusters(data:dict, size: int) -> list:
    labels = [-1] * size
    for label, keys in data.items():
        for key in keys:
            labels[key - 1] = label
    return labels
