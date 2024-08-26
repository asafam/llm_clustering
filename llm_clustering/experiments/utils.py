from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import pickle
import logging
from llms.models import LLM
from llms.utils import generate_prompt
from embedding.models import TextEmbeddingModel
from datasets import TextClusterDataset
from clustering.constraints_manager import ConstraintsType
from llms.utils import PromptType


def get_logger():
    logger = logging.getLogger('default')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def save_experiments(experiments, file_path):
    logger = logging.getLogger('default')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as file:
        pickle.dump(experiments, file)
    logger.debug(f"{len(experiments)} experiments saved to {file_path}")

    
def load_experiments(file_path):
    logger = logging.getLogger('default')
    if not os.path.exists(file_path):
        logger.error(f"Could not find experiments file in {file_path}")
        return None
    
    with open(file_path, 'rb') as file:
        experiments = pickle.load(file)
    logger.debug(f"{len(experiments)} experiments loaded from {file_path}")
    return experiments


def is_experiment_completed(experiments, **kwargs):
    for experiment in experiments:
        if all(item in experiment.items() for item in kwargs.items()):
            return True
    return False


def get_prompt_type(constraints_type: ConstraintsType):
    if constraints_type == ConstraintsType.HardLabelsConstraints:
        return PromptType.HardLabelsClusteringPrompt
    elif constraints_type == ConstraintsType.FuzzyLabelsConstraints:
        return PromptType.FuzzyLabelsClusteringPrompt
    elif constraints_type == ConstraintsType.MustLinkCannotLinkConstraints:
        return PromptType.MustLinkCannotLinkClusteringPrompt
    

def generate_constraints(llm: LLM, constraint_type: ConstraintsType, prompt: str, **kwargs):
    # generate prompt
    prompt_type = get_prompt_type(constraint_type=constraint_type)
    prompt = generate_prompt(prompt_type=prompt_type, **kwargs) # ...

    # execute prompt (possibly n times)
    results = llm.query_prompt(prompt=prompt)
    # process results and generate constraints

    if constraints_type == ConstraintsType.Hard:
        results = 


def embed(
        df: pd.DataFrame, 
        text_embedding_model: TextEmbeddingModel, 
        text_columns: str = 'text',
        label_column: str = 'label', 
        batch_size: int = 128
    ) -> pd.DataFrame:
    """
    Function to encode a batch of texts
    """
    texts = df[text_columns].tolist()
    labels = df[label_column].tolist()

    text_cluster_dataset = TextClusterDataset(texts, labels)
    dataloader = DataLoader(text_cluster_dataset, batch_size=batch_size, shuffle=True)
    
    # Iterate over the DataLoader and encode each batch
    all_embeddings = []
    all_labels = []

    for batch_texts, batch_labels in dataloader:
        batch_embeddings = text_embedding_model.embed(batch_texts)
        all_embeddings.append(batch_embeddings)
        all_labels.extend(batch_labels)

    # Convert the list of embeddings to a single numpy array
    X = np.vstack(all_embeddings)

    # Combine embeddings with intents
    embedding_label_df = pd.DataFrame(all_embeddings)
    embedding_label_df['label'] = [tensor.item() for tensor in all_labels]
    
    return embedding_label_df
