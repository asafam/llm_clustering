import os
import pickle
import boto3
import logging
from clustering.constraints_manager import ConstraintsType
from llms.utils import PromptType
from enum import Enum


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


def get_experiment_results_item_value(item):
    if isinstance(item, (int, float, str, bool, type(None))):
        return item
    
    else:
        return str(item)

def get_prompt_type(constraint_type: ConstraintsType) -> PromptType:
    if constraint_type == ConstraintsType.HardLabelsConstraints:
        return PromptType.HardLabelsClusteringPrompt
    elif constraint_type == ConstraintsType.HardLabelsExcludeUncertainConstraints:
        return PromptType.HardLabelsClusteringExcludeUncertainPrompt
    elif constraint_type == ConstraintsType.FuzzyLabelsConstraints:
        return PromptType.FuzzyLabelsClusteringPrompt
    elif constraint_type == ConstraintsType.MustLinkCannotLinkConstraints:
        return PromptType.MustLinkCannotLinkClusteringPrompt
    elif constraint_type == ConstraintsType.KConstraint:
        return PromptType.KPredictClusteringPrompt