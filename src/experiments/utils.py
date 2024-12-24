from typing import *
import os
import pickle
import boto3
import logging
from clustering.constraints_manager import ConstraintsType
from embedding.models import TextEmbeddingModel
from llms.utils import PromptType
from enum import Enum
import numpy as np
from torch.utils.data import DataLoader
import logging


def save_experiments(data, file_path, s3_bucket_name=None, s3_object_base_path=None):
    logger = logging.getLogger('default')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

    logger.debug(f"{len(data)} experiments results saved to {file_path}")

    if s3_bucket_name and s3_object_base_path:
        file_name = os.path.basename(file_path)
        s3_client = boto3.client('s3')
        s3_object_name = os.path.join(s3_object_base_path, file_name)
        s3_client.upload_file(file_path, s3_bucket_name, s3_object_name)
        logger.debug(f"Experiments file {file_path} uploaded to {s3_bucket_name}/{s3_object_base_path}")

    
def load_experiments(file_path):
    logger = logging.getLogger('default')
    if not os.path.exists(file_path):
        logger.error(f"Could not find experiments file in {file_path}")
        return None
    
    with open(file_path, 'rb') as file:
        experiments = pickle.load(file)
    logger.debug(f"{len(experiments)} experiments loaded from {file_path}")
    return experiments


def is_experiment_completed(experiments_results, excluded_keys = ['dataset'], **kwargs):
    for experiment_results in experiments_results:
        filtered_experiment_args = {key: get_experiment_results_item_value(value) for key, value in experiment_results.items() if key not in excluded_keys}
        filtered_args = {key: get_experiment_results_item_value(value) for key, value in kwargs.items() if key not in excluded_keys}
        if all((item in filtered_experiment_args.items()) for item in filtered_args.items()):
            return True
    return False

def get_experiment_results(experiments_results, excluded_keys = ['dataset'], **kwargs):
    for experiment_results in experiments_results:
        filtered_experiment_args = {key: get_experiment_results_item_value(value) for key, value in experiment_results.items() if key not in excluded_keys}
        filtered_args = {key: get_experiment_results_item_value(value) for key, value in kwargs.items() if key not in excluded_keys}
        if all((item in filtered_experiment_args.items()) for item in filtered_args.items()):
            return experiment_results
    return None


def get_experiment_results_item_value(item):
    if isinstance(item, (int, float, str, bool, type(None))):
        return item
    
    else:
        return str(item)


def setup_logger(file_path: str, worker_id: Optional[None] = None):
    # Create or retrieve the logger
    logger = logging.getLogger('default')
    
    # Check if a handler with the given file_path already exists
    file_handler_exists = any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(file_path)
        for handler in logger.handlers
    )

    # Only add a new FileHandler if none exists for this file_path
    if file_handler_exists:
        logger.debug(f"Re-using logger writing to {file_path}")
        return logger
    
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Stream Handler (for console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if worker_id is not None:
        log_format = f"[Worker %(process)d] {log_format}"
    formatter = logging.Formatter(log_format)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler (for file output)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_handler = logging.FileHandler(file_path)  # Log file name (you can specify the path)
    file_handler.setLevel(logging.DEBUG) # Set the log level for file handler
    file_handler.setFormatter(formatter) # Use the same formatter
    logger.addHandler(file_handler)
    
    logger.debug(f"Creating a new logger writing to {file_path}")

    return logger


def get_prompt_type(constraint_type: ConstraintsType) -> PromptType:
    if constraint_type == ConstraintsType.HardLabelsConstraints:
        return PromptType.HardLabelsClusteringPrompt
    elif constraint_type == ConstraintsType.HardLabelsExcludeUncertainConstraints:
        return PromptType.HardLabelsClusteringExcludeUncertainPrompt
    elif constraint_type == ConstraintsType.FuzzyLabelsConstraints:
        return PromptType.FuzzyLabelsClusteringPrompt
    elif constraint_type == ConstraintsType.MustLinkCannotLinkConstraints:
        return PromptType.MustLinkCannotLinkClusteringPrompt
    elif constraint_type == ConstraintsType.KCountConstraint:
        return PromptType.KPredictNumberClusteringPrompt
    elif constraint_type == ConstraintsType.KNameConstraint:
        return PromptType.KPredictNameClusteringPrompt
    

def encode_dataset(
        dataset,
        model: Optional[TextEmbeddingModel],
        file_path: Optional[str],
        batch_size: int = 128,
        **kwargs
    ):
     # embed the dataset for clustering
    all_ids = []
    all_embeddings = []
    all_labels = []

    embeddings = None
    if file_path is not None:
        embeddings = np.load(file_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_ids, batch_texts, batch_labels, _, _ in dataloader:
        all_labels.extend(batch_labels)
        all_ids.extend(batch_ids)
        if embeddings is not None:
            all_embeddings.append(embeddings[batch_ids])
        else:
            batch_embeddings = model.embed(batch_texts, **kwargs)
            all_embeddings.append(batch_embeddings)
    X = np.vstack(all_embeddings)
    labels_true = [tensor.item() for tensor in all_labels]
    ids = [tensor.item() for tensor in all_ids]
    
    return X, ids, labels_true


def serialize_dataset_embeddings(
        dataset,
        model: TextEmbeddingModel,
        file_path: str,
        batch_size: int = 128
):
    # Initialize an empty list to store all embeddings from each batch
    all_embeddings = []

    # Create a DataLoader for the dataset, specifying batch size and no shuffling
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Iterate over each batch in the DataLoader
    for _, batch_texts, _ in dataloader:
        # Generate embeddings for the current batch of texts
        batch_embeddings = model.embed(batch_texts)
        all_embeddings.append(batch_embeddings)

    # Stack all batch embeddings vertically to form a single numpy array
    X = np.vstack(all_embeddings)

    # Save the final embeddings matrix to the specified file path in .npy format
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, X)

