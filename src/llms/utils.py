from typing import *
from enum import Enum
import os
import yaml
import logging


class PromptType(Enum):
    SimpleClusteringPrompt = 'simple_clustering_prompt1_0'
    SimpleClusteringPrompt2 = 'simple_clustering_prompt1_1'
    HardLabelsClusteringPrompt = 'hard_labels_clustering_prompt'
    MultiStepHardLabelsClusteringPrompt = 'hard_labels_clustering_multi_step_prompt'
    MultiStepHardLabelsClusteringPromptB = 'hard_labels_clustering_multi_step_prompt_b'
    MultiStepHardLabelsClusteringPrompt2 = 'hard_labels_clustering_multi_step_prompt2'
    MultiStepHardLabelsClusteringPrompt3 = 'hard_labels_clustering_multi_step_prompt3'
    HardLabelsClusteringCoTPrompt = 'hard_labels_clustering_prompt_cot'
    HardLabelsClusteringExcludeUncertainPrompt = 'hard_labels_clustering_exclude_uncertain_prompt'
    FuzzyLabelsClusteringPrompt = 'fuzzy_labels_clustering_prompt'
    MustLinkCannotLinkClusteringPrompt = 'must_link_cannot_link_clustering_prompt'
    KPredictNameClusteringPrompt = 'k_predict_name_clustering_prompt'
    KPredictNumberClusteringPrompt = 'k_predict_number_clustering_prompt'

def load_prompts(prompt_type: PromptType):
    # Load the YAML file
    prompt_file = os.path.join(os.getenv('LLM_CLUSTERING_BASE_DIR', ''), 'prompts', f'{prompt_type.value}.yaml')
    with open(prompt_file, 'r') as file:
        data = yaml.safe_load(file)
        template_prompts = list(map(lambda d: dict(system_prompt=d.get('system_prompt'), user_prompt=d.get('user_prompt')), data))

    return template_prompts


def generate_prompt(prompt_type: PromptType, **kwargs):
    logger = logging.getLogger('default')
    # Load the YAML file
    prompt_file = os.path.join(os.getenv('LLM_CLUSTERING_BASE_DIR', ''), 'prompts', f'{prompt_type.value}.yaml')
    logger.debug(f"generate_prompt: Loading prompt file {prompt_file}")
    with open(prompt_file, 'r') as file:
        data = yaml.safe_load(file)
        template_prompts = map(lambda d: d['prompt'], data)

    template_prompt = template_prompts[0]

    # Format the texts as [ID: {index}] {text}
    ids = kwargs.get('ids')
    texts = kwargs.get('texts')
    formatted_texts = "\n".join([f"[ID: {id}] {text}" for id, text in zip(ids, texts)])
    prompt = template_prompt.replace("{texts}", formatted_texts) # Replace the {texts} placeholder

    # Format the hint
    hint = kwargs.get('hint', "their meaning")
    prompt = prompt.replace("{hint}", hint) # Replace the {hint} placeholder

    # Format the k_info 
    k = kwargs.get('k')
    prompt = prompt.replace("{k_info}\n\n", f"the goal is to partition the data into exactly {k} clusters. However, the provided sample data may contain fewer than {k} clusters due to its limited size.\n\n" if (k is not None and k > 0) else "") # Replace the {k_info} placeholder

    return prompt


def get_formatter(prompt_type: PromptType, step: int = 0) -> Callable:
    if prompt_type in [
        PromptType.SimpleClusteringPrompt2, PromptType.HardLabelsClusteringPrompt, PromptType.MultiStepHardLabelsClusteringPromptB, 
    ]:
        return format_response_as_dictionary_of_clusters
    elif prompt_type in [
        PromptType.SimpleClusteringPrompt, PromptType.HardLabelsClusteringPrompt, PromptType.HardLabelsClusteringCoTPrompt, 
        PromptType.HardLabelsClusteringExcludeUncertainPrompt, PromptType.MultiStepHardLabelsClusteringPrompt, 
        PromptType.MultiStepHardLabelsClusteringPrompt2, PromptType.MultiStepHardLabelsClusteringPrompt3
    ]:
        return format_response_as_dictionary_of_sentences
    elif prompt_type == PromptType.MustLinkCannotLinkClusteringPrompt:
        return format_response_as_must_link_cannot_link
    elif prompt_type == PromptType.KPredictNumberClusteringPrompt:
        return format_response_as_value_of_k_number
    else:
        return None


def format_response_as_dictionary_of_clusters(data: dict, context: Optional[dict] = None, **kwargs) -> list:
    sentences_labels = context.get('sentences_labels', {}) if context else {}
    for label, sids in data['result'].items():
        for sid in sids:
            sentences_labels[sid] = label
    
    explanations = data.get('explanations')

    return dict(sentences_labels=sentences_labels, explanations=explanations)


def format_response_as_dictionary_of_sentences(data: dict, context: Optional[dict] = None, **kwargs) -> list:
    sentences_labels = context.get('sentences_labels', {}) if context else {}
    for sid, label in data['result'].items():
        sentences_labels[sid] = label
    
    explanations = data.get('explanations')

    return dict(sentences_labels=sentences_labels, explanations=explanations)


def format_response_as_must_link_cannot_link(data: dict, **kwargs):
    must_link = []
    cannot_link = []

    # Flatten the must_link list
    for tup in data.get("must_link", []):
        if len(tup) == 2:
            must_link.append(tup)
        elif len(tup) > 2:
            for i in range(len(tup)):
                for j in range(i + 1, len(tup)):
                    must_link.append((tup[i], tup[j]))

    for tup in data.get("cannot_link", []):
        if len(tup) == 2:
            cannot_link.append(tup)
        elif len(tup) > 2:
            for i in range(len(tup)):
                for j in range(i + 1, len(tup)):
                    cannot_link.append((tup[i], tup[j]))
    
    return dict(
        must_link = must_link,
        cannot_link = cannot_link,
    )

def format_response_as_value_of_k_number(data: dict, **kwargs):
    return dict(
        k = int(data['result'])
    )