import os
import pickle
import boto3
import logging
from clustering.constraints_manager import ConstraintsType
from llms.utils import PromptType



def save_experiments(experiments, file_path, s3_bucket_name=None, s3_object_base_path=None):
    logger = logging.getLogger('default')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as file:
        pickle.dump(experiments, file)
    logger.debug(f"{len(experiments)} experiments saved to {file_path}")

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


def is_experiment_completed(experiments, **kwargs):
    for experiment in experiments:
        if all(item in experiment.items() for item in kwargs.items()):
            return True
    return False


def get_prompt_type(constraint_type: ConstraintsType):
    if constraint_type == ConstraintsType.HardLabelsConstraints:
        return PromptType.HardLabelsClusteringPrompt
    elif constraint_type == ConstraintsType.FuzzyLabelsConstraints:
        return PromptType.FuzzyLabelsClusteringPrompt
    elif constraint_type == ConstraintsType.MustLinkCannotLinkConstraints:
        return PromptType.MustLinkCannotLinkClusteringPrompt