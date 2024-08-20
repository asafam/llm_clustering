from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import pickle
import logging
from models.embedding import TextEmbeddingModel
from datasets import TextClusterDataset


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


def embed(
        df: pd.DataFrame, 
        text_embedding_model: TextEmbeddingModel, 
        label_column: str = 'label', 
        batch_size: int = 128
    ) -> pd.DataFrame:
    """
    Function to encode a batch of texts
    """
    texts = df['text'].tolist()
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
    all_embeddings = np.vstack(all_embeddings)

    # Combine embeddings with intents
    embedding_label_df = pd.DataFrame(all_embeddings)
    embedding_label_df['label'] = [tensor.item() for tensor in all_labels]
    
    return embedding_label_df