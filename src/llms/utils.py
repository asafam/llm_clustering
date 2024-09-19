from typing import *
from enum import Enum
import os
import yaml
import logging


OFFSET = 1

class PromptType(Enum):
    SimpleClusteringPrompt = 'simple_clustering_prompt1_0'
    SimpleClusteringPrompt2 = 'simple_clustering_prompt1_1'
    HardLabelsClusteringPrompt = 'hard_labels_clustering_prompt'
    HardLabelsClusteringCoTPrompt = 'hard_labels_clustering_prompt_cot'
    FuzzyLabelsClusteringPrompt = 'fuzzy_labels_clustering_prompt'
    MustLinkCannotLinkClusteringPrompt = 'must_link_cannot_link_clustering_prompt'
    KPredictClusteringPrompt = 'k_predict_clustering_prompt'


def generate_prompt(prompt_type: PromptType, text_index_offset: int = OFFSET, **kwargs):
    logger = logging.getLogger('default')
    # Load the YAML file
    prompt_file = os.path.join(os.getenv('LLM_CLUSTERING_BASE_DIR', ''), 'prompts', f'{prompt_type.value}.yaml')
    logger.debug(f"Loading prompt file {prompt_file}")
    with open(prompt_file, 'r') as file:
        data = yaml.safe_load(file)
        template_prompt = data[0]['prompt']

    # Format the texts as [ID: {index}] {text}
    texts = kwargs.get('texts')
    formatted_texts = "\n".join([f"[ID: {index}] {text}" for index, text in enumerate(texts, start=text_index_offset)])
    prompt = template_prompt.replace("{texts}", formatted_texts) # Replace the {texts} placeholder

    # Format the hint
    hint = kwargs.get('hint', "their meaning")
    prompt = prompt.replace("{hint}", hint) # Replace the {hint} placeholder

    # Format the k_info 
    k = kwargs.get('k')
    prompt = prompt.replace("{k_info}\n\n", f"Number of clusters: {k}\n\n" if (k is not None and k > 0) else "") # Replace the {k_info} placeholder

    return prompt


def get_formatter(prompt_type: PromptType) -> Callable:
    if prompt_type in [PromptType.SimpleClusteringPrompt, PromptType.HardLabelsClusteringPrompt, PromptType.HardLabelsClusteringCoTPrompt]:
        return format_response_as_dictionary_of_sentences
    elif prompt_type in [PromptType.MustLinkCannotLinkClusteringPrompt]:
        return format_response_as_must_link_cannot_link
    elif prompt_type == PromptType.SimpleClusteringPrompt2:
        return format_response_as_dictionary_of_clusters
    else:
        raise ValueError(f"No formatter found for {prompt_type}")


def format_response_as_dictionary_of_clusters(data: dict, size: int, text_index_offset: int = OFFSET) -> list:
    labels = [-1] * size
    for label, keys in data['result'].items():
        for key in keys:
            labels[key - text_index_offset] = label
    return labels


def format_response_as_dictionary_of_sentences(data: dict, size: int, text_index_offset: int = OFFSET) -> list:
    # labels = [-1] * size
    # for key, label in data['result'].items():
    #     labels[key - text_index_offset] = label
    # return labels
    result = {}
    for key, value in data['result'].items():
        result[key - text_index_offset] = value
    data['result'] = result
    return data


def format_response_as_must_link_cannot_link(data: dict, text_index_offset: int = OFFSET) -> list:
    must_link = [(a - text_index_offset, b - text_index_offset) for (a, b) in data['must_link']]
    data['cannot_link'] = [(a - text_index_offset, b - text_index_offset) for (a, b) in data['cannot_link']]
    return data
