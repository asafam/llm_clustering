from enum import Enum
import os
import yaml
import logging


class PromptType(Enum):
    HardLabelsClusteringPrompt = 'fuzzy_label_clustering_prompt'
    FuzzyLabelsClusteringPrompt = 'fuzzy_label_clustering_prompt'
    MustLinkCannotLinkClusteringPrompt = 'must_link_cannot_link_clustering_prompt'


def generate_prompt(prompt_type: PromptType, **kwargs):
    logger = logging.getLogger('default')
    # Load the YAML file
    prompt_file = os.path.join(os.getenv('LLM_CLUSTERING_BASE_DIR', ''), 'prompts', f'{prompt_type.value}.yaml')
    logger.debug(f"Loading prompt file {prompt_file}")
    with open(prompt_file, 'r') as file:
        template_prompt = yaml.safe_load(file)

    # Format the documents as [ID: {index}] {document}
    documents = kwargs.get('documents')
    document_index_offset = kwargs.get('document_index_offset', 1)
    formatted_documents = "\n".join([f"[ID: {index + document_index_offset}] {doc}" for index, doc in enumerate(documents, start=1)])
    prompt = template_prompt.replace("{documents}", formatted_documents) # Replace the {documents} placeholder

    # Format the k_info 
    k = kwargs.get('k')
    prompt = prompt.replace("{k_info}\n\n", f"Number of clusters: {k}\n\n" if k is not None else "") # Replace the {k_info} placeholder

    return prompt
